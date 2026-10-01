"""Run the itcmoe compression, compensation and evaluation stages sequentially."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('prepare', 'covariance', 'rank-inputs', 'rank', 'prepare-hot', 'hot', 'evaluate')
DATASETS = ('mbpp', 'gsm8k', 'multiarith', 'singleop', 'singleq')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/sdar_c10.json')
    parser.add_argument('--original-model', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--calibration-jsonl', type=Path, required=True)
    parser.add_argument('--mbpp-source-jsonl', type=Path, required=True)
    parser.add_argument('--opencompass-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--factor-library', type=Path)
    parser.add_argument('--reuse-calibration-from', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    stages, datasets = config['stages'], config['datasets']
    if config['method'] != 'itcmoe' or config['budget'] not in (10, 20, 30):
        raise ValueError('Invalid method name or budget in configuration')
    if not stages or any(stage not in STAGES for stage in stages):
        raise ValueError('The configuration contains an invalid stage')
    if [STAGES.index(stage) for stage in stages] != sorted(set(STAGES.index(stage) for stage in stages)):
        raise ValueError('Stages must be ordered and unique')
    if not datasets or len(set(datasets)) != len(datasets) or any(d not in DATASETS for d in datasets):
        raise ValueError('Dataset names must be valid, unique and nonempty')
    common = ['--budget', str(config['budget'])]
    for name in ('original_model', 'work_dir', 'calibration_jsonl', 'mbpp_source_jsonl', 'opencompass_root', 'data_dir'):
        common.extend(['--'+name.replace('_', '-'), str(getattr(args, name).resolve())])
    common.extend(['--anchor-lambda', str(config.get('anchor_lambda', .1))])
    if args.factor_library:common.extend(['--factor-library',str(args.factor_library.resolve())])
    if args.reuse_calibration_from:
        common.extend(['--reuse-calibration-from',str(args.reuse_calibration_from.resolve())])
    for stage in stages:
        if args.reuse_calibration_from and stage in ('covariance','rank-inputs'):
            continue
        for dataset in (datasets if stage == 'evaluate' else [None]):
            command = [sys.executable, str(ROOT/'run.py'), stage, *common]
            if dataset:
                command.extend(['--dataset', dataset])
            if args.dry_run:
                command.append('--dry-run')
            print(json.dumps({'stage': stage, 'dataset': dataset, 'command': command}), flush=True)
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
