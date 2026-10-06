"""Validate and benchmark the explicit tcgen05 affine composition kernel."""
import argparse
import json
from pathlib import Path

import torch
from benchmark import make_case, timing, errors
from variants import SegmentPlan, fast_gold_reference
from kernels import compose, descend
from blackwell_merge import compose_sm100
from hardware_compare import hardware_metadata, instruction_evidence, emit


class ExplicitMergePlan(SegmentPlan):
    def scan_tree(self):
        assert self.bm == self.bn == 64 and self.merge_precision == 'tf32'
        for lower, upper in zip(self.levels, self.levels[1:]):
            self.last_compose = compose_sm100[(len(upper), self.h, 8)](
                lower, upper, self.h, num_warps=8)
        previous = self.c['h0']
        for i in range(len(self.levels) - 1, -1, -1):
            self.last_descend = descend[(len(self.levels[i]), self.h, 4)](
                self.levels[i], previous, self.states[i], self.h,
                i == len(self.levels) - 1, i == 0, 64, 64, 'tf32', num_warps=8)
            previous = self.states[i]


def run():
    assert torch.cuda.get_device_capability()[0] == 10
    result = dict(metadata=hardware_metadata(), mode='explicit', rows=[], evidence={})
    rows = result['rows']
    for p, h in ((8, 12), (64, 12), (8, 96)):
        torch.manual_seed(73191)
        ab = torch.randn((p, h, 2, 128, 128), device='cuda') * .03
        ab[:, :, 0] += torch.eye(128, device='cuda') * .9
        out_auto, out_explicit = [torch.empty_like(ab[:p // 2]) for _ in range(2)]
        def auto():
            return compose[(p // 2, h, 8)](ab, out_auto, h, 64, 64, 'tf32', num_warps=8)
        def explicit():
            return compose_sm100[(p // 2, h, 8)](ab, out_explicit, h, num_warps=8)
        ak, ek = auto(), explicit()
        torch.cuda.synchronize()
        left, right = ab[0::2].double(), ab[1::2].double()
        ref = torch.stack((right[:, :, 0] @ left[:, :, 0],
                           right[:, :, 0] @ left[:, :, 1] + right[:, :, 1]), dim=2)
        delta = errors(out_explicit, ref)
        assert delta['finite'] and delta['rel_rms'] < .002, delta
        emit(rows, dict(scope='component', P=p, H=h, automatic=timing(auto),
                        explicit=timing(explicit), explicit_vs_auto=errors(out_explicit, out_auto),
                        explicit_vs_fp64=delta, auto_vs_fp64=errors(out_auto, ref)))
        if p == 8 and h == 12:
            result['evidence']['automatic'] = instruction_evidence(ak)
            result['evidence']['explicit'] = instruction_evidence(ek)
            assert result['evidence']['explicit']['ptx_static_counts']['tcgen05_mma'] > 0
            assert result['evidence']['explicit']['ptx_static_counts']['tmem_alloc'] > 0
    for t, h in ((8192, 12), (32768, 12), (8192, 96)):
        c = make_case(t, h, nonzero=True)
        base = timing(c['call'])
        for cls in (SegmentPlan, ExplicitMergePlan):
            plan = cls(c, 8, True, 64, 64, 'tf32')
            measured = timing(lambda: plan.full(True))
            emit(rows, dict(scope='full', method=cls.__name__, T=t, H=h,
                            timing=measured, original=base,
                            speedup_vs_original=base['median_ms'] / measured['median_ms'],
                            merge=timing(plan.scan_tree)))
    for gate in ('normal', 'weak', 'strong'):
        c = make_case(8192, 12, gate, True)
        go, gs = fast_gold_reference(c)
        original = SegmentPlan(c, 8, True, 64, 64, 'tf32')
        original.full(True)
        oo, os = c['out'].clone(), original.ht[-1:].clone()
        plan = ExplicitMergePlan(c, 8, True, 64, 64, 'tf32')
        plan.full(True)
        emit(rows, dict(scope='validation', gate=gate,
                        output_vs_automatic=errors(c['out'], oo),
                        state_vs_automatic=errors(plan.ht[-1:], os),
                        output_vs_fp64=errors(c['out'], go),
                        state_vs_fp64=errors(plan.ht[-1:], gs)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with torch.inference_mode():
        result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
