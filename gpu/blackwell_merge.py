"""Explicit SM100 affine composition with Gluon, TMEM and tcgen05 MMA.

FP32 inputs are consumed by the TF32 MMA instruction. This is a single-level
composition kernel; it does not implement the summary or downward scan stages.
"""
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    allocate_tensor_memory, TensorMemoryLayout, tcgen05_mma,
    fence_async_shared, mbarrier,
)


@g.jit
def compose_sm100(AB, OUT, H: gl.constexpr):
    parent, h, tile = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    kind = tile // 4
    tile = tile % 4
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 2], [1, 0])
    r = (tile // 2) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, layout))
    c = (tile % 2) * 64 + gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    ka = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    kb = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    left = (2 * parent * H + h) * 32768
    right = ((2 * parent + 1) * H + h) * 32768
    a = gl.load(AB + right + r[:, None] * 128 + ka[None, :])
    b = gl.load(AB + left + kind * 16384 + kb[:, None] * 128 + c[None, :])
    sa = gl.allocate_shared_memory(gl.float32, [64, 128],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2), a)
    sb = gl.allocate_shared_memory(gl.float32, [128, 64],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=32, rank=2, transposed=True), b)
    acc = allocate_tensor_memory(gl.float32, [64, 64], TensorMemoryLayout([64, 64], unpacked=True))
    barrier = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(barrier, count=1)
    fence_async_shared()
    tcgen05_mma(sa, sb, acc, use_acc=False, mbarriers=[barrier], mbarrier_preds=[True])
    mbarrier.wait(barrier, phase=0)
    # TMEM loads require a hardware-compatible lane mapping, distinct from
    # the coalesced global-memory layout used above.
    tmem_load_layout: gl.constexpr = gl.BlockedLayout([1, 16], [16, 2], [4, 2], [0, 1])
    value = gl.convert_layout(acc.load(tmem_load_layout), layout)
    mbarrier.invalidate(barrier)
    if kind == 1:
        value += gl.load(AB + right + 16384 + r[:, None] * 128 + c[None, :])
    gl.store(OUT + (parent * H + h) * 32768 + kind * 16384 + r[:, None] * 128 + c[None, :], value)
