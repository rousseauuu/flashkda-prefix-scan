"""Run the hardware experiments on a local CUDA GPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'gpu'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('baseline', 'correctness', 'precise', 'fast',
                        'long_validation', 'experiment', 'tune', 'confirm'), default='confirm')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from benchmark import run_suite
    result = run_suite(args.suite)
    output = args.output or ROOT / 'results' / f'{args.suite}_local.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'Results saved to {output}')


if __name__ == '__main__':
    main()
