"""Instruction-verified, same-source MMA generation comparisons."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import torch
import triton

from benchmark import errors, make_case, timing
from kernels import summary, compose


def hardware_metadata():
    p = torch.cuda.get_device_properties(0)
    return dict(gpu=p.name, capability=[p.major, p.minor], sm_count=p.multi_processor_count,
                memory_GiB=p.total_memory / 2**30, torch=str(torch.__version__),
                triton=triton.__version__, cuda=torch.version.cuda,
                disable_mma_v5=os.environ.get('DISABLE_MMA_V5', '0'),
                disable_mma_v3=os.environ.get('DISABLE_MMA_V3', '0'))


def instruction_evidence(kernel):
    ptx = kernel.asm['ptx']
    binary = kernel.asm['cubin']
    with tempfile.NamedTemporaryFile(suffix='.cubin') as f:
        f.write(binary)
        f.flush()
        dump = subprocess.run(['cuobjdump', '--dump-sass', f.name], capture_output=True,
                              text=True, check=True, timeout=60).stdout
    signatures = {'mma_sync': r'\bmma\.sync\.', 'wgmma': r'\bwgmma\.mma_async\.',
                  'tcgen05_mma': r'\btcgen05\.mma\.', 'tmem_alloc': r'\btcgen05\.alloc\.',
                  'tmem_load': r'\btcgen05\.ld\.', 'tmem_store': r'\btcgen05\.st\.'}
    counts = {name: len(re.findall(pattern, ptx)) for name, pattern in signatures.items()}
    ops = sorted(set(re.findall(r'\b(?:[A-Z]*MMA|LDTM|STTM)[A-Z0-9_.]*', dump)))
    lines = [line.strip() for line in ptx.splitlines()
             if re.search(r'\b(?:mma\.sync|wgmma\.mma_async|tcgen05\.)', line)]
    return dict(ptx_static_counts=counts, sass_matrix_instructions=ops,
                ptx_instruction_examples=list(dict.fromkeys(lines))[:24],
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
                cubin_sha256=hashlib.sha256(binary).hexdigest(),
                registers=kernel.n_regs, spills=kernel.n_spills,
                shared_bytes=kernel.metadata.shared)


def emit(rows, row):
    rows.append(row)
    print(json.dumps(row), flush=True)


def component_rows(rows, evidence, small=False):
    # CPU-generated deterministic coefficients are shared across architectures.
    # They isolate component costs; they are not an upstream K1 output fixture.
    for h, n, p in ([(12, 64, 8)] if small else [(12, 512, 8), (12, 2048, 8), (96, 512, 8)]):
        gen = torch.Generator(device='cpu').manual_seed(73191)
        def rand(shape, scale=1., dtype=torch.float32):
            return (torch.randn(shape, generator=gen) * scale).to(dtype).cuda()
        kr = rand((h, n, 16, 128), .02, torch.bfloat16)
        w = rand((h, n, 16, 128), .03)
        u = rand((h, n, 16, 128), .1)
        gt = torch.full((h, n, 128), .995, device='cuda')
        ab = torch.empty((p, h, 2, 128, 128), device='cuda')
        def summaries():
            return summary[(p, h, 8)](kr, gt, w, u, ab, h, n, n // p, True, num_warps=4)
        kernel = summaries()
        evidence.setdefault('component_summary', instruction_evidence(kernel))
        measured = timing(summaries)
        # Check summary semantics on one selected segment/head against FP64.
        a = torch.eye(128, device='cuda', dtype=torch.float64)
        b = torch.zeros_like(a)
        for i in range(n // p):
            r, wi, ui = kr[0, i].double().T, w[0, i].double(), u[0, i].double()
            a = .995 * a - r @ (wi @ a)
            b = .995 * b + r @ (ui - wi @ b)
        emit(rows, dict(scope='component', stage='summary', H=h, chunks=n, P=p,
                        timing=measured, transition_vs_fp64=errors(ab[0, 0, 0], a),
                        additive_vs_fp64=errors(ab[0, 0, 1], b)))
        # Keep a single-level binary composition identical across GPU families.
        for precision in ('tf32x3', 'tf32'):
            out = torch.empty((p // 2, h, 2, 128, 128), device='cuda')
            def merge():
                return compose[(p // 2, h, 8)](ab, out, h, 64, 64, precision, num_warps=8)
            kernel = merge()
            evidence.setdefault(f'component_merge_{precision}', instruction_evidence(kernel))
            measured = timing(merge)
            left, right = ab[0].double(), ab[1].double()
            expected_a = right[:, 0] @ left[:, 0]
            expected_b = right[:, 0] @ left[:, 1] + right[:, 1]
            emit(rows, dict(scope='component', stage='compose', H=h, chunks=n, P=p,
                            merge_precision=precision, timing=measured,
                            transition_vs_fp64=errors(out[0, :, 0], expected_a),
                            additive_vs_fp64=errors(out[0, :, 1], expected_b)))


def full_rows(rows, evidence, small=False):
    from variants import SegmentPlan, fast_gold_reference
    shapes = [(1024, 12)] if small else [(8192, 12), (32768, 12), (8192, 96)]
    for t, h in shapes:
        c = make_case(t, h, nonzero=True)
        baseline = timing(c['call'])
        c['call']()
        original = c['out'].clone()
        emit(rows, dict(scope='full', method='original', T=t, H=h, timing=baseline))
        for precision in ('tf32x3', 'tf32'):
            plan = SegmentPlan(c, 8, True, 64, 64, precision)
            measured = timing(lambda: plan.full(True))
            row = dict(scope='full', method='segment_tree', T=t, H=h, P=8,
                       merge_precision=precision, timing=measured,
                       speedup_vs_original=baseline['median_ms'] / measured['median_ms'],
                       output_vs_original=errors(c['out'], original),
                       coefficient=timing(plan.coefficients), summary=timing(plan.summaries),
                       merge=timing(plan.scan_tree), replay=timing(plan.replay))
            emit(rows, row)
            if h == 12 and t == shapes[0][0]:
                for name, kernel in [('coefficients', plan.last_coefficients),
                                     ('summary', plan.last_summary),
                                     (f'compose_{precision}', plan.last_compose),
                                     (f'descend_{precision}', plan.last_descend)]:
                    evidence.setdefault(name, instruction_evidence(kernel))
    for gate in ('normal', 'weak', 'strong'):
        c = make_case(1024 if small else 8192, 12, gate, True)
        go, gs = fast_gold_reference(c)
        c['call']()
        emit(rows, dict(scope='validation', method='original', gate=gate, T=c['t'], H=12,
                        output_vs_fp64=errors(c['out'], go), state_vs_fp64=errors(c['ht'], gs)))
        for precision in ('tf32x3', 'tf32'):
            plan = SegmentPlan(c, 8, True, 64, 64, precision)
            plan.full(True)
            emit(rows, dict(scope='validation', method='segment_tree', gate=gate,
                            T=c['t'], H=12, P=8, merge_precision=precision,
                            output_vs_fp64=errors(c['out'], go),
                            state_vs_fp64=errors(plan.ht[-1:], gs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('auto', 'legacy'), required=True)
    parser.add_argument('--components-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Callers set these before process startup, before importing Triton.
    expected = '1' if args.mode == 'legacy' else '0'
    assert os.environ.get('DISABLE_MMA_V5', '0') == expected
    assert os.environ.get('DISABLE_MMA_V3', '0') == expected
    result = dict(metadata=hardware_metadata(), mode=args.mode, rows=[], evidence={})
    print(json.dumps(result['metadata']), flush=True)
    with torch.inference_mode():
        component_rows(result['rows'], result['evidence'], args.smoke)
        if not args.components_only:
            full_rows(result['rows'], result['evidence'], args.smoke)
    if args.mode == 'legacy':
        for e in result['evidence'].values():
            assert e['ptx_static_counts']['tcgen05_mma'] == 0
            assert e['ptx_static_counts']['wgmma'] == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
