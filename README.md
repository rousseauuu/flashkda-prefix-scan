# FlashKDA Prefix Scan Experiments

An experimental study of parallelizing the inter-chunk state recurrence in [FlashKDA](https://github.com/MoonshotAI/FlashKDA) with associative affine scans.

On a single NVIDIA B200, an eight-segment tree prototype achieves **1.64x at 8K tokens** and **1.82x at 32K tokens** with 12 heads. The results include preprocessing and state replay. The same approach does **not** improve the tested 96-head cases, and a dense tree with one leaf per chunk is slower than the original kernel.

This repository contains research prototypes, numerical checks, and measured results. It is not a production replacement for FlashKDA.

## Algorithm

Using a key-by-value state layout, one chunk has the affine form

```text
residual = U - W @ S
S_next   = D @ S + R @ residual
         = A @ S + B
A        = D - R @ W
B        = R @ U
```

Here `D` is diagonal decay, and `R`, `W`, and `U` come from the chunk coefficients. The composition of a left segment followed by a right segment is

```text
(A_right, B_right) compose (A_left, B_left)
    = (A_right @ A_left, A_right @ B_left + B_right).
```

Composition is associative, but not commutative. Both the transition `A` and the additive contribution `B` are required. A reduction alone produces only the final summary; a scan also recovers incoming states for every segment.

The prototype uses four stages:

1. Run the original K1 and prepare state-independent coefficients.
2. Compute affine summaries for several contiguous segments in parallel.
3. Build a summary tree and propagate incoming states down the tree.
4. Replay the original K2 independently inside each segment.

The downward pass propagates actual states rather than materializing every full prefix transition matrix. Within each segment, K2 retains its original sequential recurrence.

## Performance

The independent confirmation run used batch size 1, 12 heads, key/value dimension 128, chunk size 16, eight segments, and nonzero initial states. The tree GEMM tile was 64 by 64.

| Tokens | Original | Tree, TF32x3 merge | Speedup | Tree, TF32 merge | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.777 ms | 0.504 ms | 1.54x | 0.474 ms | 1.64x |
| 32,768 | 3.048 ms | 1.705 ms | 1.79x | 1.674 ms | 1.82x |

Source: [`results/confirm_results.json`](results/confirm_results.json). The summary stage uses BF16 operands with FP32 accumulation and storage in both columns. TF32x3 and TF32 describe the **tree merge**, not the entire pipeline. A separate version using higher-precision summary arithmetic did not show a consistent end-to-end improvement in the initial sweep.

Other measured outcomes:

- At 8K tokens and 96 heads, the original took about 1.019 ms. The best initial segmented candidate took about 3.098 ms. Keep the original path for this tested shape.
- A direct 512-leaf tree at 8K/12-head took 2.742 ms even after increasing the merge tile and using TF32. That is about 3.5x slower than the original.
- Splitting independent value columns into separate Triton programs matched the original output exactly in the tested cases, but did not improve performance.
- SASS inspection of the upstream binary found `HMMA.16816` instructions. This study changes parallelism; it does not establish the performance of a `tcgen05` implementation.

Low head counts leave less independent work for the original inter-chunk recurrence. Segmentation trades additional summary computation and memory traffic for more parallel work. At larger head counts, the extra work can dominate. These measurements support shape-dependent dispatch, not a universal replacement.

## Numerical validation

Associativity holds for the mathematical affine recurrence. It does not make the reordered computation bitwise equivalent to the original kernel, which rounds intermediate states to BF16 at chunk boundaries.

Validation includes an independent FP64 token recurrence, normal/weak/strong decay cases, and nonzero initial states. The accelerated FP64 reference was checked against a direct PyTorch FP64 reference on a short case. CPU algebra checks additionally cover tail chunks and non-square states. Error metrics are calculated after casting both tensors to FP32.

For the eight-segment TF32 candidate under weak decay:

| Tokens | Original output relative RMS | Tree output relative RMS | Original final-state relative RMS | Tree final-state relative RMS |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.7974% | 0.7608% | 0.7376% | 0.7382% |
| 32,768 | 0.8013% | 0.7925% | 0.7349% | 0.7389% |

These are errors against the FP64 reference, not errors against the original kernel. The weak-decay input uses gate logits of -12; strong decay uses +8. Other inputs use seeded random tensors. In normal and strong decay cases, the measured errors were essentially unchanged at the reported precision.

More aggressive scanning can worsen accuracy: the 512-leaf TF32 tree reached approximately 0.982% output relative RMS in the 8K weak-decay test, versus approximately 0.797% for the original. Results do not establish model-quality equivalence or a general error bound.

## Reproduce

Measured environment:

- NVIDIA B200, compute capability 10.0, 148 SMs, approximately 178 GiB visible memory.
- Python 3.11, CUDA toolkit 12.9.1, PyTorch 2.8.0+cu129, Triton 3.4.0.
- FlashKDA commit `1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`.
- CUTLASS commit `5c149f52a436782210263fb2f19b354443a61c6a`.

Use a Linux environment with a CUDA 12.9 development toolkit, `gcc`, `g++`, and Git. The setup script targets B200 (`sm_100a`). Other architectures have not been validated.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu129
python3 -m pip install -r requirements.txt
bash scripts/setup_flashkda.sh

# Algebra only; no GPU required.
OPENBLAS_NUM_THREADS=1 python3 scripts/check_prefix_scan.py

# Check the upstream baseline and host stage selector.
python3 scripts/run_gpu.py --suite baseline
python3 scripts/run_gpu.py --suite correctness

# Reproduce the candidate confirmation and long-sequence checks.
python3 scripts/run_gpu.py --suite confirm

# Optional broader sweeps.
python3 scripts/run_gpu.py --suite experiment
python3 scripts/run_gpu.py --suite tune
```

The setup script downloads upstream FlashKDA into ignored `vendor/FlashKDA` and applies a host-side stage selector: `stage=0` runs both kernels, `stage=1` runs K1, and `stage=2` runs K2. The CUDA kernel bodies remain unchanged. Use `FLASHKDA_SOURCE` to point metadata collection at another matching, patched source checkout.

New runs write `results/*_local.json` by default, leaving the recorded data intact. Timings are medians of three CUDA Graph measurements, each targeting 80 ms. Full-pipeline timings include K1, coefficient preparation, summaries, scanning, and K2 replay. They exclude buffer allocation and most Python/host launch overhead. Compilation is outside the timed region. Separately measured stage times need not sum exactly to full-pipeline times.

The GPU experiments require dense, chunk-aligned sequences that divide evenly into the selected power-of-two segment counts. No backward pass, production variable-length integration, B300 results, or real-model quality evaluation is included. The provider-independent runner packages the measured kernels; its local CLI and CPU checks have been validated, but this packaging was not used for a new GPU timing run.

## Files

| Path | Contents |
| --- | --- |
| `gpu/kernels.py` | Triton summaries, affine composition, tree descent, value splitting, and FP64 reference |
| `gpu/variants.py` | Segmented orchestration, timing sweeps, and numerical checks |
| `gpu/benchmark.py` | Inputs, upstream baseline, timing, and reference metrics |
| `scripts/patch_flashkda.py` | Host launch-stage selector for the pinned upstream revision |
| `scripts/check_prefix_scan.py` | NumPy algebra checks and a rounding counterexample |
| `results/` | Recorded CPU, baseline, sweep, tuning, and confirmation data |

## Related work and attribution

The baseline is [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA). The affine formulation is also described in [FLA's context-parallel documentation](https://github.com/fla-org/flash-linear-attention/blob/8024667ab58fdd8986587147fca71fecc017f977/fla/ops/cp/README.md). [inclusionAI/cuLA](https://github.com/inclusionAI/cuLA/tree/79be249e61453808e18e5cef7702b363239e7d8d/cula/ops/kda) provides related KDA prescan implementations. Associative affine scanning is established prior work; the contribution here is the implementation and empirical comparison of these particular configurations against the pinned FlashKDA baseline.

Upstream projects and downloaded dependencies retain their respective licenses. This repository does not vendor their source trees.
