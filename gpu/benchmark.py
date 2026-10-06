import json
import subprocess
import time
import re
import os
from pathlib import Path


def metadata():
    import torch
    import triton
    p = torch.cuda.get_device_properties(0)
    source = Path(os.environ.get('FLASHKDA_SOURCE', Path(__file__).resolve().parents[1] / 'vendor' / 'FlashKDA'))
    return {'gpu': p.name, 'capability': [p.major, p.minor], 'sm_count': p.multi_processor_count,
            'memory_GiB': p.total_memory / 2**30, 'torch': str(torch.__version__),
            'triton': triton.__version__, 'cuda': torch.version.cuda,
            'flashkda_commit': subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
            'cutlass_commit': subprocess.check_output(['git', '-C', str(source / 'cutlass'), 'rev-parse', 'HEAD'], text=True).strip(),
            'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True)}


def make_case(t, h, gate='normal', nonzero=False):
    import torch
    from flash_kda_C import fwd, get_workspace_size
    torch.manual_seed(20261006)
    shape = (1, t, h, 128)
    q, k, v = [torch.randn(shape, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
    g = torch.randn_like(q) if gate == 'normal' else torch.full_like(q, -12.0 if gate == 'weak' else 8.0)
    beta = torch.randn(shape[:-1], device='cuda', dtype=torch.bfloat16)
    a_log = torch.zeros(h, device='cuda', dtype=torch.float32)
    bias = torch.zeros((h, 128), device='cuda', dtype=torch.float32)
    h0 = torch.randn((1, h, 128, 128), device='cuda') * 0.1 if nonzero else torch.zeros((1, h, 128, 128), device='cuda')
    ht = torch.empty_like(h0)
    out = torch.empty_like(q)
    ws = torch.empty(get_workspace_size(t, h, 1), device='cuda', dtype=torch.uint8)

    def call(stage=0):
        fwd(q, k, v, g, beta, 128**-0.5, out, ws, a_log, bias, -5.0,
            initial_state=h0, final_state=ht, stage=stage)

    return dict(t=t, h=h, q=q, k=k, v=v, g=g, beta=beta, a_log=a_log, bias=bias,
                h0=h0, ht=ht, out=out, ws=ws, call=call, gate=gate)


def timing(fn):
    import torch
    import triton.testing
    fn()
    torch.cuda.synchronize()
    samples = [triton.testing.do_bench_cudagraph(fn, rep=80) for _ in range(3)]
    return {'median_ms': sorted(samples)[1], 'samples_ms': samples}


def errors(actual, expected):
    import torch
    a, b = actual.float(), expected.float()
    delta = a - b
    return {'rel_rms': (delta.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-20)).item(),
            'max_abs': delta.abs().max().item(), 'finite': bool(torch.isfinite(a).all().item())}


def gold_reference(c):
    """Independent fp64 token recurrence from raw inputs and exact activations."""
    import torch
    q, k, v = [c[name][0].double() for name in ('q', 'k', 'v')]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * 128**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    decay = (-5 * torch.sigmoid(c['a_log'].double().exp()[None, :, None] *
                               (c['g'][0].double() + c['bias'].double()[None]))).exp()
    beta = c['beta'][0].double().sigmoid()
    state = c['h0'][0].double().transpose(-1, -2).contiguous()
    out = torch.empty_like(v)
    for i in range(c['t']):
        state *= decay[i, :, :, None]
        residual = v[i] - (k[i, :, :, None] * state).sum(-2)
        state += k[i, :, :, None] * (beta[i, :, None] * residual)[:, None, :]
        out[i] = (q[i, :, :, None] * state).sum(-2)
    return out[None], state.transpose(-1, -2)[None]


def run_suite(suite):
    import torch
    result = {'metadata': metadata(), 'suite': suite, 'rows': []}
    print(json.dumps(result['metadata']), flush=True)
    if suite == 'correctness':
        import flash_kda_C
        dump = subprocess.run(['cuobjdump', '--dump-sass', flash_kda_C.__file__], capture_output=True, text=True, timeout=60)
        result['sass'] = {'returncode': dump.returncode,
                          'matrix_instructions': sorted(set(re.findall(r'\b(?:HMMA|HGMMA|UMMA)[A-Z0-9_.]*', dump.stdout))),
                          'hmmma_occurrences': dump.stdout.count('HMMA')}
        print(json.dumps(result['sass']), flush=True)
    with torch.inference_mode():
        if suite == 'baseline':
            for gate in ('normal', 'weak', 'strong'):
                c = make_case(256, 2, gate, nonzero=True)
                go, gs = gold_reference(c)
                c['call']()
                whole_o, whole_s = c['out'].clone(), c['ht'].clone()
                c['call'](1)
                c['call'](2)
                assert torch.equal(c['out'], whole_o) and torch.equal(c['ht'], whole_s)
                row = {'type': 'correctness', 'gate': gate, 'stage_split_bitwise_equal': True,
                       'output_vs_fp64': errors(c['out'], go), 'state_vs_fp64': errors(c['ht'], gs)}
                result['rows'].append(row)
                print(json.dumps(row), flush=True)
            for t, h in ((8192, 12), (8192, 96), (32768, 12), (32768, 96)):
                c = make_case(t, h)
                c['call']()
                row = {'type': 'timing', 'T': t, 'H': h, 'full': timing(c['call']),
                       'K1': timing(lambda: c['call'](1)), 'K2': timing(lambda: c['call'](2))}
                result['rows'].append(row)
                print(json.dumps(row), flush=True)
        else:
            from variants import run_variants
            result['rows'] = run_variants(suite)
    return result
