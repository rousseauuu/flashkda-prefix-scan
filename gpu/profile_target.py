"""Warm one pipeline stage, then expose one launch to Nsight Compute."""
import argparse
import json

import torch

from benchmark import make_case
from kernels import compose, descend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('k2', 'summary', 'compose', 'compose_sm100',
                        'descend', 'coefficients', 'replay'), required=True)
    parser.add_argument('--tokens', type=int, default=8192)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--segments', type=int, default=8)
    parser.add_argument('--precision', choices=('tf32', 'tf32x3'), default='tf32')
    parser.add_argument('--tile', type=int, choices=(16, 32, 64), default=64)
    parser.add_argument('--merge-warps', type=int, default=0)
    parser.add_argument('--summary-width', type=int, choices=(32, 64, 128), default=32)
    parser.add_argument('--summary-warps', type=int, default=4)
    args = parser.parse_args()
    with torch.inference_mode():
        c = make_case(args.tokens, args.heads, nonzero=True)
        from profile_tuning import SummaryTilePlan
        plan = SummaryTilePlan(c, args.segments, args.summary_width, args.summary_warps,
                               args.tile, args.precision)
        tiles = (128 // args.tile) ** 2
        warps = args.merge_warps or (8 if args.tile == 64 else 4)
        plan.full(True)
        if args.stage == 'k2':
            fn = lambda: c['call'](2)
        elif args.stage == 'summary':
            fn = plan.summaries
        elif args.stage == 'coefficients':
            fn = plan.coefficients
        elif args.stage == 'replay':
            fn = plan.replay
        elif args.stage == 'compose':
            fn = lambda: compose[(args.segments // 2, args.heads, 2 * tiles)](
                plan.ab, plan.levels[1], args.heads, args.tile, args.tile,
                args.precision, num_warps=warps)
        elif args.stage == 'compose_sm100':
            assert args.precision == 'tf32' and args.tile == 64
            from blackwell_merge import compose_sm100
            fn = lambda: compose_sm100[(args.segments // 2, args.heads, 8)](
                plan.ab, plan.levels[1], args.heads, num_warps=8)
        else:
            fn = lambda: descend[(args.segments, args.heads, tiles)](
                plan.ab, plan.states[1], plan.carry, args.heads, False, True,
                args.tile, args.tile, args.precision, num_warps=warps)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        print(json.dumps(vars(args)), flush=True)
        torch.cuda.cudart().cudaProfilerStart()
        fn()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    main()
