"""Export numeric counters from an NCU raw CSV without host/process metadata."""
import argparse
import csv
import json
from pathlib import Path

PREFIXES = ('gpu__', 'sm__', 'smsp__', 'l1tex__', 'lts__', 'dram__', 'launch__', 'derived__')


def counters(text):
    rows = list(csv.reader(text.splitlines()))
    if len(rows) != 3:
        raise ValueError('Expected header, units, and exactly one profiled launch')
    result = {}
    for name, unit, value in zip(*rows):
        if name.startswith(PREFIXES):
            try:
                number = float(value.replace(',', ''))
            except ValueError:
                continue
            result[name] = dict(unit=unit, value=number)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(counters(args.csv.read_text()), indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
