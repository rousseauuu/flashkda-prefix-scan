"""CPU algebra check for FlashKDA chunk and segment affine scans.

Run: OPENBLAS_NUM_THREADS=1 python check_prefix_scan.py
Requires NumPy only. This is not a CUDA benchmark or an emulation of all
FlashKDA bf16 rounding, approximate exponentials, and fp16 inverse operations.

Equations follow the public FLA KDA token recurrence and
FlashKDA/tests/torch_ref.py. State layout here is [key_dim, value_dim].
"""

import json
from pathlib import Path

import numpy as np


def relative_l2(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def generate_inputs(length, dim, value_dim, gate_mode, seed):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(length, dim)) for _ in range(2))
    q /= np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6)
    k /= np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
    q /= np.sqrt(dim)
    v = rng.normal(size=(length, value_dim))
    if gate_mode == "weak":
        g = -rng.uniform(1e-6, 1e-4, size=k.shape)
    elif gate_mode == "strong":
        g = -rng.uniform(1.0, 5.0, size=k.shape)
    elif gate_mode == "mixed":
        g = -10.0 ** rng.uniform(-5.0, np.log10(5), size=k.shape)
    else:
        g = -rng.uniform(0.01, 0.15, size=k.shape)
    beta = rng.uniform(0, 1, size=length)
    beta[::19] = 0
    beta[::23] = 1
    state0 = rng.normal(size=(dim, value_dim)) * 0.2
    return q, k, v, g, beta, state0


def token_reference(q, k, v, g, beta, state0, chunk_size):
    state = state0.copy()
    outputs, boundaries = [], []
    for t in range(len(q)):
        if t % chunk_size == 0:
            boundaries.append(state.copy())
        decayed = np.exp(g[t])[:, None] * state
        state = decayed + beta[t] * np.outer(k[t], v[t] - k[t] @ decayed)
        outputs.append(q[t] @ state)
    return np.stack(outputs), np.stack(boundaries), state


def prepare_chunks(q, k, v, g, beta, chunk_size, dtype):
    length, dim = q.shape
    count = (length + chunk_size - 1) // chunk_size
    padded_length = count * chunk_size

    def pad_chunk(x):
        result = np.zeros((padded_length, *x.shape[1:]), dtype=dtype)
        result[:length] = x
        return result.reshape(count, chunk_size, *x.shape[1:])

    q, k, v, g, beta = map(pad_chunk, (q, k, v, g, beta))
    cumulative = np.cumsum(g, axis=1)
    kd = k * np.exp(cumulative)
    qd = q * np.exp(cumulative)
    ki = k * np.exp(-cumulative)
    kr = k * np.exp(cumulative[:, -1:, :] - cumulative)
    decay = np.exp(cumulative[:, -1, :])
    lower = beta[..., None] * np.tril(kd @ ki.swapaxes(-1, -2), -1)
    eye = np.eye(chunk_size, dtype=dtype)[None, :, :]
    rhs = beta[..., None] * np.concatenate((kd, v), axis=-1)
    wu = np.linalg.solve(eye + lower, rhs)
    w, u = wu[..., :dim], wu[..., dim:]
    mqk = np.tril(qd @ ki.swapaxes(-1, -2))
    return decay, kr.swapaxes(-1, -2), w, u, qd, mqk


def chunk_step(chunks, i, state):
    decay, kt, w, u, qd, mqk = chunks
    residual = u[i] - w[i] @ state
    out = qd[i] @ state + mqk[i] @ residual
    next_state = decay[i, :, None] * state + kt[i] @ residual
    return out, next_state


def serial_chunks(chunks, state0):
    state = state0.copy()
    outputs, states = [], []
    for i in range(len(chunks[0])):
        states.append(state.copy())
        out, state = chunk_step(chunks, i, state)
        outputs.append(out)
    return np.concatenate(outputs), np.stack(states), state


def chunk_summaries(chunks):
    decay, kt, w, u, _, _ = chunks
    dim = w.shape[-1]
    a = decay[:, :, None] * np.eye(dim, dtype=w.dtype)[None, :, :] - kt @ w
    b = kt @ u
    return a, b


def tree_incoming(a, b, state0):
    """Work-efficient upsweep plus state-only downsweep; preserves time order.

    Each batch matmul at one level is independent. The CPU execution checks
    algebra; it does not model GPU scheduling, communication or latency.
    """
    count, dim = a.shape[:2]
    padded_count = 1 << (count - 1).bit_length()
    ap = np.broadcast_to(np.eye(dim, dtype=a.dtype), (padded_count, dim, dim)).copy()
    bp = np.zeros((padded_count, *b.shape[1:]), dtype=b.dtype)
    ap[:count], bp[:count] = a, b
    levels = [(ap, bp)]
    while len(ap) > 1:
        left_a, right_a = ap[::2], ap[1::2]
        left_b, right_b = bp[::2], bp[1::2]
        ap, bp = right_a @ left_a, right_a @ left_b + right_b
        levels.append((ap, bp))
    final = ap[0] @ state0 + bp[0]
    states = state0[None].copy()
    for child_a, child_b in reversed(levels[:-1]):
        next_states = np.empty((len(child_a), *state0.shape), dtype=state0.dtype)
        next_states[::2] = states
        next_states[1::2] = child_a[::2] @ states + child_b[::2]
        states = next_states
    return states[:count], final


def outputs_from_incoming(chunks, states):
    _, _, w, u, qd, mqk = chunks
    return (qd @ states + mqk @ (u - w @ states)).reshape(-1, u.shape[-1])


def segment_summaries(chunks, chunks_per_segment):
    """Build segment maps without materializing a dense map per chunk."""
    decay, kt, w, u, _, _ = chunks
    dim, value_dim = w.shape[-1], u.shape[-1]
    summaries_a, summaries_b, starts = [], [], []
    for start in range(0, len(decay), chunks_per_segment):
        a = np.eye(dim, dtype=w.dtype)
        b = np.zeros((dim, value_dim), dtype=w.dtype)
        for i in range(start, min(start + chunks_per_segment, len(decay))):
            a = decay[i, :, None] * a - kt[i] @ (w[i] @ a)
            b = decay[i, :, None] * b + kt[i] @ (u[i] - w[i] @ b)
        starts.append(start)
        summaries_a.append(a)
        summaries_b.append(b)
    return np.stack(summaries_a), np.stack(summaries_b), starts


def segmented_replay(chunks, state0, chunks_per_segment):
    a, b, starts = segment_summaries(chunks, chunks_per_segment)
    incoming, final = tree_incoming(a, b, state0)
    outputs, states = [], []
    for segment, start in enumerate(starts):
        state = incoming[segment]
        for i in range(start, min(start + chunks_per_segment, len(chunks[0]))):
            states.append(state.copy())
            out, state = chunk_step(chunks, i, state)
            outputs.append(out)
    return np.concatenate(outputs), np.stack(states), final


def round_bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).astype(np.float64)


def rounding_counterexample():
    # Even rounding only at chunk boundaries destroys exact affine composition.
    # Half an ulp at 1.0 rounds to even; two half ulps add to one full ulp.
    a = np.array([[[1.0]], [[1.0]]])
    b = np.array([[[1 / 256]], [[1 / 256]]])
    state0 = np.array([[1.0]])
    sequential = state0.copy()
    for ai, bi in zip(a, b):
        sequential = round_bf16(ai @ sequential + bi)
    fused = round_bf16((a[1] @ a[0]) @ state0 + a[1] @ b[0] + b[1])
    assert not np.array_equal(sequential, fused)
    return {"rounded_each_step": sequential.item(), "rounded_after_compose": fused.item()}


def run_case(length, dim, value_dim, gate_mode, seed):
    inputs = generate_inputs(length, dim, value_dim, gate_mode, seed)
    reference_o, reference_s, reference_final = token_reference(*inputs, chunk_size=16)
    records = []
    for dtype in (np.float64, np.float32):
        chunks = prepare_chunks(*inputs[:-1], chunk_size=16, dtype=dtype)
        state0 = inputs[-1].astype(dtype)
        serial_o, serial_s, serial_final = serial_chunks(chunks, state0)
        a, b = chunk_summaries(chunks)
        tree_s, tree_final = tree_incoming(a, b, state0)
        tree_o = outputs_from_incoming(chunks, tree_s)
        segment_o, segment_s, segment_final = segmented_replay(chunks, state0, chunks_per_segment=32)
        for method, out, states, final in (
            ("serial_chunk", serial_o, serial_s, serial_final),
            ("full_tree", tree_o, tree_s, tree_final),
            ("segment_tree", segment_o, segment_s, segment_final),
        ):
            out = out[:length]
            record = dict(T=length, K=dim, V=value_dim, gate=gate_mode, dtype=np.dtype(dtype).name, method=method)
            record.update(output_max_abs=float(np.max(np.abs(out - reference_o))),
                          output_rel_l2=relative_l2(out, reference_o),
                          incoming_rel_l2=relative_l2(states, reference_s),
                          final_rel_l2=relative_l2(final, reference_final))
            tolerance = 1e-11 if dtype == np.float64 else 1e-4
            for actual, expected in ((out, reference_o), (states, reference_s), (final, reference_final)):
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
            records.append(record)
    return records


def main():
    cases = [(1, 8, 5, "moderate"), (83, 32, 24, "mixed"), (521, 32, 32, "strong"),
             (1024, 128, 128, "moderate"), (8192, 128, 128, "weak")]
    results = []
    for seed, case in enumerate(cases):
        records = run_case(*case, seed=seed)
        results.extend(records)
        tree = [r for r in records if r["method"] == "full_tree"]
        print(f"PASS T={case[0]} K={case[1]} V={case[2]} gate={case[3]} "
              + " ".join(f"{r['dtype']}_output_max_abs={r['output_max_abs']:.3e}" for r in tree), flush=True)
    counterexample = rounding_counterexample()
    print("bf16 boundary-rounding counterexample:", counterexample)
    path = Path(__file__).resolve().parents[1] / "results" / "cpu_algebra_local.json"
    path.write_text(json.dumps({"checks": results, "rounding_counterexample": counterexample,
                               "scope": "CPU algebra only; not a GPU benchmark or complete bf16 kernel emulation"}, indent=2) + "\n")
    print(f"Saved {len(results)} comparisons to {path}")


if __name__ == "__main__":
    main()
