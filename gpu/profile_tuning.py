"""Small, profiler-motivated controls for merge spilling and summary reuse."""
import argparse
import json
from pathlib import Path

import torch
from benchmark import make_case, timing, errors
from variants import SegmentPlan, fast_gold_reference
from kernels import summary, compose, descend
from hardware_compare import hardware_metadata, instruction_evidence, emit


class SummaryTilePlan(SegmentPlan):
    def __init__(self, c, p, bv, warps, tile=64, precision='tf32'):
        super().__init__(c, p, True, tile, tile, precision)
        self.bv, self.summary_warps = bv, warps

    def summaries(self):
        self.last_summary = summary[(self.p, self.h, 2 * (128 // self.bv))](
            self.ws['kr'], self.ws['gt'], self.w, self.u, self.ab,
            self.h, self.n, self.l, True, self.bv, num_warps=self.summary_warps)


class MergeTilePlan(SegmentPlan):
    def __init__(self, c, tile, warps):
        super().__init__(c, 8, True, tile, tile, 'tf32x3')
        self.merge_warps = warps

    def scan_tree(self):
        tiles = (128 // self.bm) * (128 // self.bn)
        for lower, upper in zip(self.levels, self.levels[1:]):
            self.last_compose = compose[(len(upper), self.h, 2 * tiles)](
                lower, upper, self.h, self.bm, self.bn, self.merge_precision,
                num_warps=self.merge_warps)
        previous = self.c['h0']
        for i in range(len(self.levels) - 1, -1, -1):
            self.last_descend = descend[(len(self.levels[i]), self.h, tiles)](
                self.levels[i], previous, self.states[i], self.h,
                i == len(self.levels) - 1, i == 0, self.bm, self.bn,
                self.merge_precision, num_warps=self.merge_warps)
            previous = self.states[i]


def run(mode):
    result = dict(metadata=hardware_metadata(), mode=mode, rows=[], evidence={})
    rows, evidence = result['rows'], result['evidence']
    if mode == 'confirm':
        for t in (8192, 32768):
            for gate in ('normal', 'weak', 'strong'):
                c = make_case(t, 12, gate, True)
                go, gs = fast_gold_reference(c)
                c['call']()
                base = timing(c['call']) if gate == 'normal' else None
                emit(rows, dict(scope='validation', method='original', T=t, gate=gate,
                                output_vs_fp64=errors(c['out'], go),
                                state_vs_fp64=errors(c['ht'], gs), timing=base))
                for p, bv, warps in [(8, 32, 4), (16, 64, 8)]:
                    plan = SummaryTilePlan(c, p, bv, warps)
                    plan.full(True)
                    row = dict(scope='validation', method='segmented', T=t, gate=gate,
                               P=p, BV=bv, warps=warps,
                               output_vs_fp64=errors(c['out'], go), state_vs_fp64=errors(plan.ht[-1:], gs))
                    if base:
                        row['timing'] = timing(lambda: plan.full(True))
                        row['speedup_vs_original'] = base['median_ms'] / row['timing']['median_ms']
                    emit(rows, row)
        return result
    if mode == 'legacy':
        configs = [(32, 4), (32, 8), (64, 8), (64, 16), (16, 4)]
        for t in (8192,):
            c = make_case(t, 12, nonzero=True)
            for tile, warps in configs:
                plan = MergeTilePlan(c, tile, warps)
                measured = timing(lambda: plan.full(True))
                emit(rows, dict(scope='merge_control', T=t, H=12, tile=tile, warps=warps,
                                timing=measured, merge=timing(plan.scan_tree)))
                if t == 8192:
                    for k, v in [('compose', plan.last_compose), ('descend', plan.last_descend)]:
                        evidence[f'{k}_{tile}_w{warps}'] = instruction_evidence(v)
        c = make_case(8192, 12, 'weak', True)
        go, gs = fast_gold_reference(c)
        for tile, warps in configs:
            plan = MergeTilePlan(c, tile, warps)
            plan.full(True)
            emit(rows, dict(scope='validation', gate='weak', tile=tile, warps=warps,
                            output_vs_fp64=errors(c['out'], go), state_vs_fp64=errors(plan.ht[-1:], gs)))
        return result
    configs = [(8, 32, 4), (8, 64, 4), (8, 64, 8), (8, 128, 8),
               (16, 32, 4), (16, 64, 8), (16, 128, 8)]
    for t in (8192, 32768):
        c = make_case(t, 12, nonzero=True)
        base = timing(c['call'])
        for p, bv, warps in configs:
            plan = SummaryTilePlan(c, p, bv, warps)
            measured = timing(lambda: plan.full(True))
            emit(rows, dict(scope='summary_control', T=t, H=12, P=p, BV=bv, warps=warps,
                            timing=measured, summary=timing(plan.summaries), original=base,
                            speedup_vs_original=base['median_ms'] / measured['median_ms'],
                            registers=plan.last_summary.n_regs, spills=plan.last_summary.n_spills,
                            shared_bytes=plan.last_summary.metadata.shared))
            if t == 8192:
                evidence[f'summary_p{p}_v{bv}_w{warps}'] = instruction_evidence(plan.last_summary)
    for gate in ('normal', 'weak', 'strong'):
        c = make_case(8192, 12, gate, True)
        go, gs = fast_gold_reference(c)
        for p, bv, warps in configs:
            plan = SummaryTilePlan(c, p, bv, warps)
            plan.full(True)
            emit(rows, dict(scope='validation', gate=gate, T=8192, H=12, P=p, BV=bv, warps=warps,
                            output_vs_fp64=errors(c['out'], go), state_vs_fp64=errors(plan.ht[-1:], gs)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('auto', 'legacy', 'confirm'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        result = run(args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
