import json
import time
import torch
from flash_kda_C import fwd
from benchmark import make_case, timing, errors, gold_reference
from kernels import prepare_wu, summary, serial_merge, compose, descend, k2_value_split, gold_recurrent


def workspace_views(c):
    n, h = c['t'] // 16, c['h']
    count = n * h
    offset = 0
    result = {}
    for key, elements, dtype, shape in (
        ('kd', 2048, torch.bfloat16, (h, n, 16, 128)),
        ('qd', 2048, torch.bfloat16, (h, n, 16, 128)),
        ('kr', 2048, torch.bfloat16, (h, n, 16, 128)),
        ('gt', 128, torch.float32, (h, n, 128)),
        ('inv', 256, torch.bfloat16, (h, n, 16, 16)),
        ('mqk', 256, torch.bfloat16, (h, n, 16, 16)),
    ):
        size = count * elements * (4 if dtype == torch.float32 else 2)
        result[key] = c['ws'][offset:offset + size].view(dtype).view(shape)
        offset += size
    return result


class SegmentPlan:
    def __init__(self, c, p, fast=False, bm=32, bn=32, merge_precision='tf32x3'):
        self.c, self.p, self.fast = c, p, fast
        self.bm, self.bn, self.merge_precision = bm, bn, merge_precision
        self.n, self.h = c['t'] // 16, c['h']
        assert self.n % p == 0 and p >= 2 and p & (p - 1) == 0
        self.l = self.n // p
        self.ws = workspace_views(c)
        self.w = torch.empty((self.h, self.n, 16, 128), device='cuda', dtype=torch.float32)
        self.u = torch.empty_like(self.w)
        self.ab = torch.empty((p, self.h, 2, 128, 128), device='cuda', dtype=torch.float32)
        self.carry = torch.empty((p, self.h, 128, 128), device='cuda', dtype=torch.float32)
        self.ht = torch.empty_like(self.carry)
        self.args = [c[key].reshape(p, c['t'] // p, self.h, 128) for key in ('q', 'k', 'v', 'g')]
        self.beta = c['beta'].reshape(p, c['t'] // p, self.h)
        self.out = c['out'].reshape(p, c['t'] // p, self.h, 128)
        self.levels = [self.ab]
        size = p
        while size > 2:
            size //= 2
            self.levels.append(torch.empty((size, self.h, 2, 128, 128), device='cuda', dtype=torch.float32))
        self.states = [torch.empty((len(level), self.h, 128, 128), device='cuda', dtype=torch.float32)
                       for level in self.levels]
        self.states[0] = self.carry

    def coefficients(self):
        self.last_coefficients = prepare_wu[(self.h, self.n)](self.ws['kd'], self.ws['inv'], self.c['v'], self.c['beta'],
                                     self.w, self.u, self.h, self.n, num_warps=4)

    def summaries(self):
        self.last_summary = summary[(self.p, self.h, 8)](self.ws['kr'], self.ws['gt'], self.w, self.u, self.ab,
                                     self.h, self.n, self.l, self.fast, num_warps=4)

    def scan_serial(self):
        serial_merge[(self.h, 4)](self.ab, self.c['h0'], self.carry, self.h, self.p,
                                  PREC=self.merge_precision, num_warps=4)

    def scan_tree(self):
        tiles = (128 // self.bm) * (128 // self.bn)
        warps = 8 if self.bm * self.bn >= 4096 else 4
        for lower, upper in zip(self.levels, self.levels[1:]):
            self.last_compose = compose[(len(upper), self.h, 2 * tiles)](
                lower, upper, self.h, self.bm, self.bn, self.merge_precision, num_warps=warps)
        previous = self.c['h0']
        for i in range(len(self.levels) - 1, -1, -1):
            self.last_descend = descend[(len(self.levels[i]), self.h, tiles)](
                self.levels[i], previous, self.states[i], self.h,
                i == len(self.levels) - 1, i == 0, self.bm, self.bn, self.merge_precision, num_warps=warps)
            previous = self.states[i]

    def replay(self):
        fwd(*self.args, self.beta, 128**-0.5, self.out, self.c['ws'], self.c['a_log'], self.c['bias'], -5.,
            initial_state=self.carry, final_state=self.ht, stage=2)

    def full(self, tree=False):
        self.c['call'](1)
        self.coefficients()
        self.summaries()
        self.scan_tree() if tree else self.scan_serial()
        self.replay()


def split_call(c, bv, full=True):
    ws = workspace_views(c)

    def call():
        if full:
            c['call'](1)
        k2_value_split[(c['h'], 128 // bv)](
            ws['kd'], ws['qd'], ws['kr'], ws['gt'], ws['inv'], ws['mqk'],
            c['v'], c['beta'], c['h0'], c['ht'], c['out'], c['h'], c['t'] // 16, bv, num_warps=4,
            enable_fp_fusion=False)
    return call


def correctness_rows():
    rows = []
    for gate in ('normal', 'weak', 'strong'):
        c = make_case(512, 2, gate, nonzero=True)
        go, gs = gold_reference(c)
        c['call']()
        bo, bs = c['out'].clone(), c['ht'].clone()
        baseline = {'method': 'original', 'gate': gate, 'output_vs_fp64': errors(bo, go), 'state_vs_fp64': errors(bs, gs)}
        rows.append(baseline)
        print(json.dumps(baseline), flush=True)
        for fast in (False, True):
            plan = SegmentPlan(c, 4, fast)
            for tree in (False, True):
                plan.full(tree)
                row = {'method': 'segment_tree' if tree else 'segment_serial', 'P': 4, 'fast': fast, 'gate': gate,
                       'output_vs_fp64': errors(c['out'], go), 'state_vs_fp64': errors(plan.ht[-1:], gs),
                       'output_vs_original': errors(c['out'], bo)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                assert row['output_vs_fp64']['finite'] and row['state_vs_fp64']['finite']
        for bv in (32, 64, 128):
            split_call(c, bv)()
            row = {'method': 'value_split', 'BV': bv, 'gate': gate, 'output_vs_fp64': errors(c['out'], go),
                   'state_vs_fp64': errors(c['ht'], gs), 'output_vs_original': errors(c['out'], bo)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def performance_rows(shapes, fast=False):
    rows = []
    for t, h in shapes:
        c = make_case(t, h)
        baseline = timing(c['call'])
        bo = c['out'].clone()
        row = {'T': t, 'H': h, 'method': 'original', 'timing': baseline}
        rows.append(row)
        print(json.dumps(row), flush=True)
        for bv in (32, 64, 128):
            fn = split_call(c, bv)
            measured = timing(fn)
            row = {'T': t, 'H': h, 'method': 'value_split', 'BV': bv, 'timing': measured,
                   'speedup': baseline['median_ms'] / measured['median_ms'], 'output_vs_original': errors(c['out'], bo)}
            rows.append(row)
            print(json.dumps(row), flush=True)
        for p in (4, 16, 64):
            plan = SegmentPlan(c, p, fast)
            plan.full()
            for tree in (False, True):
                fn = lambda tree=tree: plan.full(tree)
                measured = timing(fn)
                row = {'T': t, 'H': h, 'method': 'segment_tree' if tree else 'segment_serial', 'P': p, 'fast': fast,
                       'timing': measured, 'speedup': baseline['median_ms'] / measured['median_ms'],
                       'output_vs_original': errors(c['out'], bo),
                       'coefficient': timing(plan.coefficients), 'summary': timing(plan.summaries),
                       'merge': timing(plan.scan_tree if tree else plan.scan_serial), 'replay': timing(plan.replay)}
                rows.append(row)
                print(json.dumps(row), flush=True)
            del plan
        # Direct per-chunk tree for the small-head 8K workload.
        if t == 8192 and h == 12:
            plan = SegmentPlan(c, t // 16, fast)
            measured = timing(lambda: plan.full(True))
            row = {'T': t, 'H': h, 'method': 'full_chunk_tree', 'P': t // 16, 'fast': fast,
                   'timing': measured, 'speedup': baseline['median_ms'] / measured['median_ms'],
                   'output_vs_original': errors(c['out'], bo), 'merge': timing(plan.scan_tree)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del plan
        del c
    return rows


def run_variants(suite):
    if suite == 'experiment':
        rows = []
        for phase in ('precise', 'fast', 'long_validation'):
            print('START_PHASE ' + phase, flush=True)
            batch = run_variants(phase)
            for row in batch:
                row['phase'] = phase
            rows.extend(batch)
        return rows
    if suite == 'correctness':
        return correctness_rows()
    if suite == 'precise':
        return performance_rows(((8192, 12), (8192, 96), (32768, 12), (32768, 96)))
    if suite == 'fast':
        return performance_rows(((8192, 12), (8192, 96), (32768, 12), (32768, 96)), fast=True)
    if suite == 'long_validation':
        return long_validation()
    if suite == 'tune':
        return tune_merge()
    if suite == 'confirm':
        return confirm_candidates()
    raise ValueError(suite)


def fast_gold_reference(c):
    q, k, v = [c[name].double() for name in ('q', 'k', 'v')]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * 128**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    decay = (-5 * torch.sigmoid(c['a_log'].double().exp()[None, None, :, None] *
                               (c['g'].double() + c['bias'].double()[None, None]))).exp()
    beta, h0 = c['beta'].double().sigmoid(), c['h0'].double()
    out, ht = torch.empty_like(v), torch.empty_like(h0)
    gold_recurrent[(c['h'], 8)](q, k, v, decay, beta, h0, ht, out, c['h'], c['t'], num_warps=4)
    return out, ht


def long_validation():
    rows = []
    c = make_case(64, 2, 'weak', True)
    py_o, py_s = gold_reference(c)
    tr_o, tr_s = fast_gold_reference(c)
    torch.testing.assert_close(tr_o, py_o, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(tr_s, py_s, atol=1e-12, rtol=1e-12)
    rows.append({'gold_gpu_vs_pytorch_validated': True})
    for gate in ('normal', 'weak', 'strong'):
        c = make_case(8192, 12, gate, True)
        go, gs = fast_gold_reference(c)
        c['call']()
        methods = [('original', c['call'], c['ht'])]
        methods += [(f'value_split_{bv}', split_call(c, bv), c['ht']) for bv in (32, 64, 128)]
        plans = []
        for fast in (False, True):
            for p in (4, 16, 64, 512):
                plan = SegmentPlan(c, p, fast)
                plans.append(plan)
                methods.append((f'segment_tree_p{p}_fast{fast}', lambda plan=plan: plan.full(True), plan.ht[-1:]))
        for name, fn, state in methods:
            fn()
            row = {'method': name, 'gate': gate, 'T': 8192, 'H': 12,
                   'output_vs_fp64': errors(c['out'], go), 'state_vs_fp64': errors(state, gs)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def tune_merge():
    rows = []
    for t, h in ((8192, 12), (32768, 12), (8192, 96)):
        c = make_case(t, h)
        base = timing(c['call'])
        for p in (8, 16, 32, 64):
            for bm, precision in ((64, 'tf32x3'), (64, 'tf32'), (128, 'tf32')):
                plan = SegmentPlan(c, p, True, bm, bm, precision)
                measured = timing(lambda: plan.full(True))
                row = {'phase': 'tuned_performance', 'T': t, 'H': h, 'P': p, 'tile': bm, 'merge_precision': precision,
                       'fast': True, 'timing': measured, 'speedup': base['median_ms'] / measured['median_ms'],
                       'original': base, 'merge': timing(plan.scan_tree), 'summary': timing(plan.summaries),
                       'registers': plan.last_compose.n_regs, 'spills': plan.last_compose.n_spills,
                       'shared_bytes': plan.last_compose.metadata.shared}
                rows.append(row)
                print(json.dumps(row), flush=True)
                del plan
    # Include the direct, one-leaf-per-chunk version with the larger merge tile.
    c = make_case(8192, 12)
    base = timing(c['call'])
    for bm, precision in ((64, 'tf32x3'), (128, 'tf32')):
        plan = SegmentPlan(c, 512, True, bm, bm, precision)
        measured = timing(lambda: plan.full(True))
        row = {'phase': 'tuned_full_tree', 'T': 8192, 'H': 12, 'P': 512, 'tile': bm,
               'merge_precision': precision, 'timing': measured,
               'speedup': base['median_ms'] / measured['median_ms'], 'original': base, 'merge': timing(plan.scan_tree)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    for gate in ('weak', 'normal'):
        c = make_case(8192, 12, gate, True)
        go, gs = fast_gold_reference(c)
        for bm, precision in ((64, 'tf32x3'), (64, 'tf32'), (128, 'tf32')):
            for p in (16, 64, 512):
                plan = SegmentPlan(c, p, True, bm, bm, precision)
                plan.full(True)
                row = {'phase': 'tuned_validation', 'gate': gate, 'T': 8192, 'H': 12, 'P': p,
                       'tile': bm, 'merge_precision': precision, 'output_vs_fp64': errors(c['out'], go),
                       'state_vs_fp64': errors(plan.ht[-1:], gs)}
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def confirm_candidates():
    rows = []
    for t in (8192, 32768):
        for gate in ('normal', 'weak', 'strong'):
            c = make_case(t, 12, gate, True)
            go, gs = fast_gold_reference(c)
            c['call']()
            base = timing(c['call']) if gate == 'normal' else None
            row = {'T': t, 'H': 12, 'gate': gate, 'method': 'original',
                   'output_vs_fp64': errors(c['out'], go), 'state_vs_fp64': errors(c['ht'], gs)}
            if base:
                row['timing'] = base
            rows.append(row)
            print(json.dumps(row), flush=True)
            for p in (8, 16, 32):
                for precision in ('tf32x3', 'tf32'):
                    plan = SegmentPlan(c, p, True, 64, 64, precision)
                    plan.full(True)
                    row = {'T': t, 'H': 12, 'gate': gate, 'method': 'segment_tree',
                           'P': p, 'tile': 64, 'merge_precision': precision,
                           'output_vs_fp64': errors(c['out'], go),
                           'state_vs_fp64': errors(plan.ht[-1:], gs)}
                    if base:
                        row['timing'] = timing(lambda: plan.full(True))
                        row['speedup'] = base['median_ms'] / row['timing']['median_ms']
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    return rows
