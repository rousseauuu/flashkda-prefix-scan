"""Profile one warmed stage; requires Nsight Compute and GPU counter access."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('auto', 'legacy'), default='auto')
    parser.add_argument('--ncu', default='ncu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', required=True, choices=('k2', 'replay', 'summary',
                        'coefficients', 'compose', 'compose_sm100', 'descend'))
    args, target_args = parser.parse_known_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['DISABLE_MMA_V3'] = env['DISABLE_MMA_V5'] = '1' if args.backend == 'legacy' else '0'
    env['TRITON_CACHE_DIR'] = str(args.output.parent.resolve() / ('cache-' + args.backend))
    kernel = {'k2': '_flash_kda_fwd_recurrence', 'replay': '_flash_kda_fwd_recurrence',
              'coefficients': 'prepare_wu'}.get(args.stage, args.stage)
    command = [args.ncu, '--target-processes', 'all', '--profile-from-start', 'off',
               '--kernel-name-base', 'function', '--kernel-name', 'regex:' + kernel,
               '--launch-count', '1', '--clock-control', 'none', '--cache-control', 'none',
               '--export', str(args.output)]
    for section in ('SpeedOfLight', 'LaunchStats', 'Occupancy', 'SchedulerStats',
                    'WarpStateStats', 'MemoryWorkloadAnalysis', 'ComputeWorkloadAnalysis'):
        command += ['--section', section]
    command += ['--metrics', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
                sys.executable, str(Path(__file__).resolve().parents[1] / 'gpu/profile_target.py'),
                '--stage', args.stage, *target_args]
    subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
