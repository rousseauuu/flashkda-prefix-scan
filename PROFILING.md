# Nsight Compute: where SM100 helps and what still limits performance

B200 hardware counters show two distinct constraints: too little parallelism in the original recurrent kernel, and latency/register pressure in the segmented summary. Native SM100 matrix instructions eliminate substantial spill traffic in the fixed TF32x3 merge configuration, but changing instructions alone does not deliver a large end-to-end improvement for every precision and shape.

## Method and evidence

Nsight Compute 2025.2.1.0 successfully collected counters on a 148-SM B200 with CUDA 12.9, PyTorch 2.8.0 and Triton 3.4.0. The study includes 14 stage/backend profiles and seven follow-up profiles. Each target is warmed ten times and a single launch is selected with the CUDA profiler API. Sections cover throughput, launch resources, occupancy, schedulers, warp stalls, memory and compute. The follow-up explicitly collects local-memory load/store sectors.

The default case is batch 1, 8,192 tokens, 12 heads, K=V=128, chunk size 16, eight segments and nonzero initial state. Additional original-K2 and summary profiles use 96 heads. Automatic and legacy lowering run in separate processes/cache directories; legacy sets both `DISABLE_MMA_V3=1` and `DISABLE_MMA_V5=1`. Original CUDA K2 is unchanged by these Triton switches. See [hardware migration](HARDWARE_MIGRATION.md) for instruction evidence and precision controls.

GPU clocks and caches are uncontrolled (`none`). Profiled durations include effects of counter replay and are **not** the performance benchmark. All speedups below come from separate CUDA Graph measurements (three samples, 80 ms target per sample). No PC sampling was collected, so stall attribution to a particular source load remains unverified.

Sanitized numeric counters, configurations and raw-report SHA-256 hashes are in [full profiles](results/ncu_b200_full.json) and [follow-up profiles](results/ncu_b200_followup.json). Raw reports are retained locally because they include host/process metadata.

## Original K2 has too little parallelism

At 12 heads the original K2 launches only 12 CTAs on 148 SMs. Its tensor-pipeline active percentage is 2.50%, with DRAM active cycles at 2.06%. Increasing to 96 heads raises tensor-pipeline activity to 20.13%, while profiled K2 duration stays roughly 729–754 microseconds. Segment replay at 12 heads raises tensor-pipeline activity to 17.66% and reduces its profiled duration to 107 microseconds, but adds summary and scan work.

These counters support parallelizing the recurrence. They do not support describing this case as saturated on HBM bandwidth. Tensor-pipeline activity is an active-cycle metric, **not percentage of peak FLOP/s**. Occupancy percentages below are averaged over active SM cycles and must be read alongside grid size.

## The summary is now the main target

For the default automatic summary:

| Metric | Value |
| --- | ---: |
| Registers per thread | 144 |
| Theoretical occupancy | 18.75% |
| Achieved occupancy | 16.05% |
| Eligible warps per scheduler per cycle | 0.275 |
| Tensor-pipeline active cycles | 5.56% |
| DRAM active cycles | 10.36% |
| Long-scoreboard stalled cycles per issued instruction | 5.50 |

The register limit permits only three resident CTAs. NCU reports low compute and memory activity and substantial waits for L1TEX dependencies. Together with the dependent chunk loop, this points to latency hiding and resource pressure. It does not establish which individual load causes the stalls, nor imply that every long-scoreboard stall is an HBM access.

Unprofiled summary time is about 0.250 ms out of 0.473 ms for the full forward. The explicit SM100 upward merge previously saved only about 1.6 microseconds for the measured merge stage, explaining its small end-to-end impact. The summary, including state movement and repeated loads, deserves priority over further tuning that small merge alone.

## SM100 removes spill traffic in the fixed TF32x3 merge

The automatic path uses `tcgen05`/TMEM. With 64x64 tiles and eight warps, compiler evidence reports 255 registers and 144 spill slots for legacy composition versus 152 registers and zero spills for native composition. The follow-up counters confirm actual local-memory traffic:

| Composition backend/configuration | Local load sectors | Local store sectors |
| --- | ---: | ---: |
| Legacy, tile 64, 8 warps | 1,953,792 | 2,171,236 |
| Legacy, tile 32, 4 warps | 1,105,920 | 1,200,296 |
| Legacy, tile 16, 4 warps | 0 | 0 |
| Native, tile 64, 8 warps | 0 | 0 |

These are L1TEX local-memory sectors, not DRAM bytes. Native tree descent also has zero local sectors; legacy tile-64 descent has 1,216,512 load and 1,355,312 store sectors. This establishes a concrete storage/resource advantage for the native configuration.

It is important to retune the legacy control too. At 8K/12 heads/TF32x3, legacy full-forward time improves from 0.614 ms with tile 64 to 0.555 ms with tile 32. Tile 16 removes spills but takes 0.572 ms; more CTAs and different tiling introduce other costs. Sixteen warps at tile 64 perform worse, at 1.342 ms. Thus the earlier native result of about 0.506 ms is approximately 1.10x faster than this retuned legacy control, rather than the roughly 1.23x fixed-configuration comparison. These are separate benchmark runs, not a claim of exhaustive optimal tuning.

For ordinary TF32, earlier automatic lowering was not faster than forced legacy lowering. SM100 benefit depends on precision, shape, register allocation and orchestration; instruction availability alone is insufficient.

## Profiler-guided summary tuning and independent confirmation

A small sweep changes segment count, summary width and warp count. The selected configuration uses 16 segments, summary width 64 and eight warps. The merge stays at tile 64. At 8K, follow-up profiling shows 128 registers/thread, 22.76% achieved occupancy and 0.415 eligible warps per scheduler, versus 144, 16.05% and 0.275 previously. Long-scoreboard stalls remain substantial (5.76 cycles per issued instruction); this is a partial improvement, not elimination of the bottleneck.

The initial sweep gave 1.675 -> 1.540 ms at 32K. A separate confirmation run produced:

| Tokens / heads | Original full forward | Previous segmented configuration | Selected configuration | Selected vs original |
| --- | ---: | ---: | ---: | ---: |
| 8,192 / 12 | 0.7761 ms | 0.4774 ms | 0.4720 ms | 1.64x |
| 32,768 / 12 | 3.0400 ms | 1.6865 ms | 1.5494 ms | 1.96x |

At 32K this is an 8.1% time reduction over the previous segmented configuration. The approximately 2x result combines algorithmic parallelism and tuning; it must not be attributed solely to an SM100 ISA migration. This selection is exposed by `SummaryTilePlan` and remains experimental rather than replacing the general default.

Both configurations were checked against the independent FP64 recurrence at 8K and 32K for normal, weak and strong gates. All output/state metrics are finite. Selected output relative RMS error ranges from 0.00526 to 0.00782; state relative RMS error ranges from 0.00456 to 0.00737, comparable to the original on these cases. This is measured numerical behavior, not bitwise equivalence, an arbitrary-input error guarantee, or real-model quality validation.

Data: [tuning sweep and legacy controls](results/profile_tuning_b200.json), [independent confirmation](results/profile_confirmation_b200.json).

## Reproduction

First follow the README's GPU setup, including the pinned and patched FlashKDA build. Nsight Compute must be installed and hardware counter access enabled. On B200:

```bash
python scripts/run_ncu.py --backend auto --stage summary --output /tmp/ncu/summary
python scripts/run_ncu.py --backend auto --stage summary --segments 16 --summary-width 64 --summary-warps 8 --output /tmp/ncu/summary_tuned
python scripts/run_ncu.py --backend legacy --stage compose --precision tf32x3 --tile 64 --output /tmp/ncu/compose_legacy
python scripts/run_ncu.py --backend auto --stage compose --precision tf32x3 --tile 64 --output /tmp/ncu/compose_native
ncu --import /tmp/ncu/compose_native.ncu-rep --page raw --csv > /tmp/ncu/compose_native.csv
python scripts/summarize_ncu.py /tmp/ncu/compose_native.csv --output /tmp/ncu/counters.json
ncu --import /tmp/ncu/compose_native.ncu-rep --page details --print-rule-details
DISABLE_MMA_V3=0 DISABLE_MMA_V5=0 TRITON_CACHE_DIR=/tmp/tuning-auto python gpu/profile_tuning.py --mode auto --output /tmp/tuning-auto.json
DISABLE_MMA_V3=1 DISABLE_MMA_V5=1 TRITON_CACHE_DIR=/tmp/tuning-legacy python gpu/profile_tuning.py --mode legacy --output /tmp/tuning-legacy.json
DISABLE_MMA_V3=0 DISABLE_MMA_V5=0 TRITON_CACHE_DIR=/tmp/tuning-auto python gpu/profile_tuning.py --mode confirm --output /tmp/confirmation.json
```

Future experiments should target summary load reuse, prefetch/software pipelining, and retaining suitable state in TMEM. A/B summary fusion could reduce duplicated operand loads but may increase live state and register pressure. These are hypotheses requiring implementation and counters; no speedup is claimed for them. This profiling study covers B200, not a new NCU comparison across all three GPU generations.
