# Hardware Migration: Ampere, Hopper, and Blackwell

This follow-up separates algorithmic parallelism from GPU backend selection. It also implements an explicit Blackwell affine-composition kernel using Gluon, shared memory, TMEM, and `tcgen05_mma`.

The original B200 prototype was already using compiler-generated tcgen05 instructions. The earlier study had inspected the upstream FlashKDA binary, but had not inspected the newly generated Triton kernels. The instruction evidence here closes that gap; it does not claim that tcgen05 first appeared only after this follow-up.

## Experimental controls

The compiler comparison uses identical kernel source, inputs, segment counts, tile sizes, and requested arithmetic precision on each GPU. Separate processes and compilation caches select:

- `auto`: `DISABLE_MMA_V3=0`, `DISABLE_MMA_V5=0`.
- `legacy`: `DISABLE_MMA_V3=1`, `DISABLE_MMA_V5=1`.

These are internal Triton compiler controls, tested with Triton 3.4.0. PTX inspection verifies their effect. They change backend lowering, including layouts and synchronization; this is not a claim that exactly one assembly instruction was replaced while every other instruction stayed fixed.

The controls are defined in [Triton's MMA support checks](https://github.com/triton-lang/triton/blob/v3.4.0/lib/Analysis/Utility.cpp). The explicit implementation uses the pinned [Gluon Blackwell interface](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/experimental/gluon/language/nvidia/blackwell/__init__.py).

Full-pipeline cases use eight segments, 64x64 merge tiles, BF16 summary operands with FP32 accumulation/storage, nonzero initial states, and the same original K1/K2. TF32 and TF32x3 are compared separately. A difference between those precision settings is not counted as an instruction-generation speedup.

Timings use three 80 ms CUDA Graph measurements and report the median. Compilation, allocation, and most host launch overhead are excluded. Component validation uses FP64 matrix operations. Full-pipeline validation covers normal, weak, and strong decay against an independent FP64 token recurrence.

## Confirmed instruction selection

| GPU | Automatic summary | Automatic merge | Forced legacy |
| --- | --- | --- | --- |
| A100-SXM4-80GB, SM80, 108 SMs | `mma.sync` | `mma.sync` | `mma.sync` |
| H100 80GB HBM3, SM90, 132 SMs | `mma.sync` + WGMMA | WGMMA | `mma.sync` |
| B200, SM100, 148 SMs | `mma.sync` + tcgen05 | tcgen05 + TMEM | `mma.sync` |

Coefficient preparation remains on `mma.sync` in the inspected full-pipeline cases. The matrix shapes inside the summary are different, so one summary kernel can contain both old and new MMA instructions.

Evidence files include PTX opcode counts and instruction examples, cubin/PTX hashes, register counts, shared-memory usage, and selected SASS mnemonics. Counts are **static occurrences**, not executed instruction counts. SASS confirms `HMMA` on the legacy path, `HGMMA` on Hopper, and `UTCHMMA` in the explicit Blackwell comparison. The first B200 collection used a narrower SASS-name filter that omitted `UTCHMMA`; its PTX still directly records tcgen05 and TMEM instructions.

## Same-GPU full-pipeline comparison

All rows below have 12 heads. The ratio is **legacy time / native time** for the same algorithm and requested precision, not speedup over upstream FlashKDA.

| GPU | Tokens | Merge precision | Forced legacy | Native automatic | Legacy/native |
| --- | ---: | --- | ---: | ---: | ---: |
| B200 | 8,192 | TF32x3 | 0.620 ms | 0.506 ms | 1.23x |
| B200 | 32,768 | TF32x3 | 1.800 ms | 1.715 ms | 1.05x |
| B200 | 8,192 | TF32 | 0.463 ms | 0.476 ms | 0.97x |
| B200 | 32,768 | TF32 | 1.643 ms | 1.686 ms | 0.97x |
| H100 | 8,192 | TF32x3 | 0.779 ms | 0.588 ms | 1.32x |
| H100 | 32,768 | TF32x3 | 2.135 ms | 1.950 ms | 1.09x |
| H100 | 8,192 | TF32 | 0.549 ms | 0.550 ms | 1.00x |
| H100 | 32,768 | TF32 | 1.909 ms | 1.911 ms | 1.00x |

For context, the upstream baseline in these runs was approximately 0.776/3.036 ms on B200 and 0.856/3.266 ms on H100 at 8K/32K. The segmented algorithm remains slower than upstream in the tested 96-head cases.

The TF32x3 tree scan illustrates where backend selection matters. On B200 at 8K/12-head, the complete scan stage took approximately 0.192 ms with legacy MMA versus 0.066 ms with native lowering. Conversely, the summary took approximately 0.242 ms with legacy MMA versus 0.253 ms with native lowering. A blanket switch to the newest instruction family is therefore not uniformly beneficial.

The B200 normal/weak/strong validation metrics matched between automatic and legacy modes for each requested merge precision. For example, weak-decay output relative RMS was 0.76075% for the TF32 segmented variant in both modes, versus 0.79742% for upstream. Equal aggregate error metrics do not prove bitwise equality between all backend outputs.

## Ampere scope and cross-GPU component measurements

The pinned upstream FlashKDA uses SM80-generation MMA **and SM90 TMA/data-movement facilities**. It is not an A100-compatible implementation merely because its matrix instructions belong to the SM80 generation. No A100 upstream end-to-end result is claimed.

Instead, all three GPUs run identical synthetic coefficient fixtures generated with a fixed CPU RNG seed. These fixtures isolate summary and affine-composition costs; they are not recorded outputs from upstream K1. The table uses 12 heads, 512 chunks, eight segments, and a single binary composition level:

| GPU, automatic backend | Summary | TF32x3 composition | TF32 composition |
| --- | ---: | ---: | ---: |
| A100 | 457.65 us | 77.27 us | 15.26 us |
| H100 | 229.31 us | 23.85 us | 10.97 us |
| B200 | 252.74 us | 20.54 us | 9.10 us |

The A100 automatic/legacy control produced the same instruction families and essentially identical timings: 457.649/457.661 us for the summary. Cross-GPU differences also reflect SM counts, clocks, bandwidth, and resource limits; they do not isolate the causal benefit of an ISA generation.

## Explicit SM100 composition

[`gpu/blackwell_merge.py`](gpu/blackwell_merge.py) implements a 64x64-tiled affine composition with a reduction dimension of 128. It explicitly allocates shared-memory operands and a TMEM accumulator, issues `tcgen05_mma`, waits on an mbarrier, reads TMEM using a compatible lane layout, and adds the affine bias where required.

The first multiply uses `use_acc=False`. In the inspected binary this eliminates the initial TMEM clearing store present in the automatic version: the automatic kernel has one static `tcgen05.st` occurrence, while the explicit kernel has none. Both have 16 static `tcgen05.mma` occurrences. Their register counts also differ, so the timing difference is not attributed exclusively to eliminating that store.

On B200, one composition level with eight segments and 12 heads took **9.13 us automatically versus 7.55 us explicitly**, a 1.21x component speedup. The component outputs matched exactly in all three tested configurations: `(P,H) = (8,12), (64,12), (8,96)`.

Only the upward composition stage was replaced in the full pipeline. Downward state propagation, summaries, coefficient preparation, and K1/K2 stayed on their existing paths. In the paired 8K/12-head run, full latency changed from **0.4728 to 0.4696 ms**, approximately **0.7% lower**. At 32K it changed from 1.6742 to 1.6711 ms, approximately 0.2% lower. These small full-pipeline differences are reported without claiming a robust production-level speedup. Normal/weak/strong validation outputs and final states matched the automatic variant exactly in these tested cases.

This is an explicit SM100 port of **one composition kernel**, not a complete rewrite of FlashKDA or a persistent, TMA-pipelined implementation of the entire scan.

A second controlled run forced all ordinary Triton stages to legacy MMA and changed only upward composition to the explicit SM100 kernel. PTX/SASS confirmed `mma.sync`/`HMMA` in the reference composition and tcgen05/`UTCHMMA` in the replacement. At `(P,H)=(8,12)`, composition changed from **8.32 to 7.56 us**, a 1.10x component speedup. Full 8K/12-head latency changed from **0.4639 to 0.4628 ms**, only about 0.24% lower. At 32K it changed from 1.6444 to 1.6435 ms. Component outputs and the tested normal/weak/strong full outputs and final states matched exactly. These measurements show a working targeted migration, while also showing that this stage is too small to deliver a large overall improvement by itself.

In the explicit comparison JSON files, the `automatic` fields identify the ordinary `tl.dot` reference. For `hardware_b200_explicit_legacy.json`, that reference is compiled with the legacy switches enabled, as recorded in `metadata`; it is not the native automatic backend.

## Reproduction

Use the versions from the main README. `cuobjdump` must be available on `PATH`.

```bash
# B200: build the baseline, then compare automatic and forced legacy lowering.
bash scripts/setup_flashkda.sh
python3 scripts/run_hardware.py

# H100: run in a separate checkout/environment with the SM90a baseline.
FLASH_KDA_CUDA_ARCHS=90a bash scripts/setup_flashkda.sh
python3 scripts/run_hardware.py

# A100: install Python dependencies, but do not build the upstream extension.
python3 scripts/run_hardware.py --components-only

# Explicit B200 composition versus the automatic compiler implementation.
DISABLE_MMA_V3=0 DISABLE_MMA_V5=0 \
  python3 gpu/explicit_sm100_compare.py --output results/explicit_sm100_local.json

# Keep other stages on legacy MMA and replace only upward composition.
DISABLE_MMA_V3=1 DISABLE_MMA_V5=1 \
  python3 gpu/explicit_sm100_compare.py --output results/explicit_legacy_local.json
```

The explicit Gluon operation remains tcgen05 even when the automatic MMA-v5 selection pass is disabled. The evidence records the instruction family of both compared composition kernels.

## Recorded data

- [`results/hardware_a100.json`](results/hardware_a100.json)
- [`results/hardware_h100.json`](results/hardware_h100.json)
- [`results/hardware_b200.json`](results/hardware_b200.json)
- [`results/hardware_b200_explicit.json`](results/hardware_b200_explicit.json)
- [`results/hardware_b200_explicit_legacy.json`](results/hardware_b200_explicit_legacy.json)

These results support selective, measured migration. The remaining larger opportunities are summary-state layout and lifetime, reducing round trips between registers/shared memory/TMEM, and scan-stage fusion. They are hypotheses for future work, not measured improvements in this repository.
