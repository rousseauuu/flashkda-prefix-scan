"""Run isolated compiler-backend comparisons on the current CUDA device."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--components-only', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import torch
    capability = torch.cuda.get_device_capability()
    components = args.components_only or capability[0] < 9
    root = Path(__file__).resolve().parents[1]
    modes = ('auto', 'legacy')
    results = []
    with tempfile.TemporaryDirectory(prefix='kda-hardware-') as tmp:
        for mode in modes:
            env = os.environ.copy()
            env['DISABLE_MMA_V3'] = env['DISABLE_MMA_V5'] = '1' if mode == 'legacy' else '0'
            env['TRITON_CACHE_DIR'] = str(Path(tmp) / f'cache-{mode}')
            output = Path(tmp) / f'{mode}.json'
            cmd = [sys.executable, str(root / 'gpu/hardware_compare.py'), '--mode', mode,
                   '--output', str(output)]
            if components:
                cmd.append('--components-only')
            if args.smoke:
                cmd.append('--smoke')
            subprocess.run(cmd, env=env, check=True)
            results.append(json.loads(output.read_text()))
    output = args.output or root / 'results' / f'hardware_sm{capability[0]}{capability[1]}_local.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'runs': results}, indent=2) + '\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
