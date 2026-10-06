"""Experimental KDA scans over unchanged FlashKDA K1 workspace.

Dense fixed-length inputs, K=V=128, C=16; all segment boundaries C-aligned.
Transition/state summary arithmetic is selectable; original segment K2 is reused.
"""
import triton
import triton.language as tl


@triton.jit
def sigmoid_match(x):
    y = tl.inline_asm_elementwise('tanh.approx.f32 $0, $1;', constraints='=f,f',
                                 args=[x * 0.5], dtype=tl.float32, is_pure=True, pack=1)
    return y * 0.5 + 0.5


@triton.jit
def prepare_wu(KD, INV, V, BETA, W, U, H: tl.constexpr, N: tl.constexpr):
    h, chunk = tl.program_id(0), tl.program_id(1)
    ht = h * N + chunk
    c = tl.arange(0, 16)
    d = tl.arange(0, 128)
    beta = sigmoid_match(tl.load(BETA + (chunk * 16 + c) * H + h).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    inv = tl.load(INV + ht * 256 + c[:, None] * 16 + c[None, :]).to(tl.float32)
    f = inv * beta[None, :]
    kd = tl.load(KD + ht * 2048 + c[:, None] * 128 + d[None, :]).to(tl.float32)
    v = tl.load(V + ((chunk * 16 + c[:, None]) * H + h) * 128 + d[None, :]).to(tl.float32)
    w = tl.dot(f, kd, input_precision='tf32x3')
    u = tl.dot(f, v, input_precision='tf32x3')
    off = ht * 2048 + c[:, None] * 128 + d[None, :]
    tl.store(W + off, w)
    tl.store(U + off, u)


@triton.jit
def summary(KR, GT, W, U, AB, H: tl.constexpr, N: tl.constexpr, L: tl.constexpr,
            FAST: tl.constexpr, BV: tl.constexpr = 32):
    seg, h, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kind = block // (128 // BV)  # 0: transition, 1: zero-input state
    r = tl.arange(0, 128)
    v = (block % (128 // BV)) * BV + tl.arange(0, BV)
    c = tl.arange(0, 16)
    state = tl.where((r[:, None] == v[None, :]) & (kind == 0), 1., 0.)
    for i in range(L):
        ht = h * N + seg * L + i
        if i == 0:
            if kind == 0:
                residual = -tl.load(W + ht * 2048 + c[:, None] * 128 + v[None, :])
            else:
                residual = tl.load(U + ht * 2048 + c[:, None] * 128 + v[None, :])
        else:
            w = tl.load(W + ht * 2048 + c[:, None] * 128 + r[None, :])
            if FAST:
                ws = tl.dot(w.to(tl.bfloat16), state.to(tl.bfloat16))
            else:
                ws = tl.dot(w, state, input_precision='tf32x3')
            if kind == 0:
                residual = -ws
            else:
                u = tl.load(U + ht * 2048 + c[:, None] * 128 + v[None, :])
                residual = u - ws
        kr = tl.load(KR + ht * 2048 + c[None, :] * 128 + r[:, None])
        decay = tl.load(GT + ht * 128 + r)
        if FAST:
            update = tl.dot(kr, residual.to(tl.bfloat16))
        else:
            update = tl.dot(kr.to(tl.float32), residual, input_precision='tf32x3')
        state = tl.fma(decay[:, None], state, update)
    off = ((seg * H + h) * 2 + kind) * 16384 + r[:, None] * 128 + v[None, :]
    tl.store(AB + off, state)


@triton.jit
def serial_merge(AB, H0, CARRY, H: tl.constexpr, P: tl.constexpr, BV: tl.constexpr = 32,
                 PREC: tl.constexpr = 'tf32x3'):
    h, bv = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    v = bv * BV + tl.arange(0, BV)
    state = tl.load(H0 + h * 16384 + v[None, :] * 128 + k[:, None]).to(tl.float32)
    for seg in range(P):
        tl.store(CARRY + (seg * H + h) * 16384 + v[None, :] * 128 + k[:, None], state)
        if seg < P - 1:
            base = (seg * H + h) * 32768
            a = tl.load(AB + base + k[:, None] * 128 + k[None, :])
            b = tl.load(AB + base + 16384 + k[:, None] * 128 + v[None, :])
            state = tl.dot(a, state, input_precision=PREC) + b


@triton.jit
def compose(AB, OUT, H: tl.constexpr, BM: tl.constexpr = 32, BN: tl.constexpr = 32,
            PREC: tl.constexpr = 'tf32x3'):
    parent, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    kind = tile // ((128 // BM) * (128 // BN))
    tile = tile % ((128 // BM) * (128 // BN))
    r = (tile // (128 // BN)) * BM + tl.arange(0, BM)
    c = (tile % (128 // BN)) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    left = ((2 * parent) * H + h) * 32768
    right = ((2 * parent + 1) * H + h) * 32768
    ra = tl.load(AB + right + r[:, None] * 128 + k[None, :])
    lb = tl.load(AB + left + kind * 16384 + k[:, None] * 128 + c[None, :])
    value = tl.dot(ra, lb, input_precision=PREC)
    if kind == 1:
        value += tl.load(AB + right + 16384 + r[:, None] * 128 + c[None, :])
    tl.store(OUT + (parent * H + h) * 32768 + kind * 16384 + r[:, None] * 128 + c[None, :], value)


@triton.jit
def descend(CHILD_AB, PARENT_S, CHILD_S, H: tl.constexpr, FIRST: tl.constexpr,
            LAST: tl.constexpr, BM: tl.constexpr = 32, BN: tl.constexpr = 32,
            PREC: tl.constexpr = 'tf32x3'):
    child, h, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    r = (tile // (128 // BN)) * BM + tl.arange(0, BM)
    c = (tile % (128 // BN)) * BN + tl.arange(0, BN)
    k = tl.arange(0, 128)
    parent = child // 2
    base = (parent * H + h) * 16384
    if child % 2 == 0:
        if FIRST:
            state = tl.load(PARENT_S + base + c[None, :] * 128 + r[:, None])
        else:
            state = tl.load(PARENT_S + base + r[:, None] * 128 + c[None, :])
    else:
        abbase = ((child - 1) * H + h) * 32768
        a = tl.load(CHILD_AB + abbase + r[:, None] * 128 + k[None, :])
        b = tl.load(CHILD_AB + abbase + 16384 + r[:, None] * 128 + c[None, :])
        if FIRST:
            incoming = tl.load(PARENT_S + base + c[None, :] * 128 + k[:, None])
        else:
            incoming = tl.load(PARENT_S + base + k[:, None] * 128 + c[None, :])
        state = tl.dot(a, incoming, input_precision=PREC) + b
    target = (child * H + h) * 16384
    if LAST:
        tl.store(CHILD_S + target + c[None, :] * 128 + r[:, None], state)
    else:
        tl.store(CHILD_S + target + r[:, None] * 128 + c[None, :], state)


@triton.jit
def k2_value_split(KD, QD, KR, GT, INV, MQK, V, BETA, H0, HT, O,
                   H: tl.constexpr, N: tl.constexpr, BV: tl.constexpr):
    h, block = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    v = block * BV + tl.arange(0, BV)
    c = tl.arange(0, 16)
    state = tl.load(H0 + h * 16384 + v[None, :] * 128 + k[:, None]).to(tl.bfloat16)
    for chunk in range(N):
        ht = h * N + chunk
        kd = tl.load(KD + ht * 2048 + c[:, None] * 128 + k[None, :])
        qd = tl.load(QD + ht * 2048 + c[:, None] * 128 + k[None, :])
        original = tl.load(V + ((chunk * 16 + c[:, None]) * H + h) * 128 + v[None, :])
        beta = sigmoid_match(tl.load(BETA + (chunk * 16 + c) * H + h).to(tl.float32)).to(tl.bfloat16)
        predicted = tl.dot(kd, state).to(tl.bfloat16)
        residual = (original.to(tl.float32) - predicted.to(tl.float32)).to(tl.bfloat16)
        residual = (residual.to(tl.float32) * beta[:, None].to(tl.float32)).to(tl.bfloat16)
        inv = tl.load(INV + ht * 256 + c[:, None] * 16 + c[None, :])
        u = tl.dot(inv, residual).to(tl.bfloat16)
        mqk = tl.load(MQK + ht * 256 + c[:, None] * 16 + c[None, :])
        out = tl.dot(qd, state).to(tl.bfloat16).to(tl.float32) + tl.dot(mqk, u).to(tl.bfloat16).to(tl.float32)
        tl.store(O + ((chunk * 16 + c[:, None]) * H + h) * 128 + v[None, :], out)
        kr = tl.load(KR + ht * 2048 + c[None, :] * 128 + k[:, None])
        update = tl.dot(kr, u)
        decay = tl.load(GT + ht * 128 + k)
        state = tl.fma(state.to(tl.float32), decay[:, None], update).to(tl.bfloat16)
    tl.store(HT + h * 16384 + v[None, :] * 128 + k[:, None], state.to(tl.float32))


@triton.jit
def gold_recurrent(Q, K, V, DECAY, BETA, H0, HT, O, H: tl.constexpr, T: tl.constexpr, BV: tl.constexpr = 16):
    h, vb = tl.program_id(0), tl.program_id(1)
    k = tl.arange(0, 128)
    v = vb * BV + tl.arange(0, BV)
    state = tl.load(H0 + h * 16384 + v[None, :] * 128 + k[:, None])
    for t in range(T):
        q = tl.load(Q + (t * H + h) * 128 + k)
        key = tl.load(K + (t * H + h) * 128 + k)
        value = tl.load(V + (t * H + h) * 128 + v)
        decay = tl.load(DECAY + (t * H + h) * 128 + k)
        beta = tl.load(BETA + t * H + h)
        decayed = state * decay[:, None]
        residual = value - tl.sum(key[:, None] * decayed, axis=0)
        state = decayed + key[:, None] * (beta * residual)[None, :]
        output = tl.sum(q[:, None] * state, axis=0)
        tl.store(O + (t * H + h) * 128 + v, output)
    tl.store(HT + h * 16384 + v[None, :] * 128 + k[:, None], state)
