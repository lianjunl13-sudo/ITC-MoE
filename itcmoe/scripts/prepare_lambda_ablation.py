"""Reuse fixed calibration inputs in separate gain-fitting directories for the anchor-coefficient ablation."""
import argparse
import json
import os
from pathlib import Path
import shutil

def copy_immutable(source, target):
    if Path(source).suffix in ('.pt', '.safetensors'):
        try:
            os.link(source, target)
            return target
        except OSError:
            pass
    return shutil.copy2(source, target)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-work', type=Path, required=True)
    p.add_argument('--output-work', type=Path, required=True)
    a = p.parse_args()
    source, target = a.source_work.resolve(), a.output_work.resolve()
    required = ['reference', 'base_model', 'model', 'hot/initial', 'hot/round1', 'hot/round2']
    for name in required:
        if not (source/name).is_dir():
            raise ValueError('Missing input directory: '+name)
    if not (source/'hot/round2/complete.json').is_file():
        raise ValueError('Second-round sampling is incomplete')
    if target.exists():
        raise ValueError('The output directory already exists; refusing to overwrite')
    for name in required:
        shutil.copytree(source/name, target/name, copy_function=copy_immutable)
    for name in ('hot/prompts.json', 'run_identity.json'):
        if (source/name).exists():
            shutil.copy2(source/name, target/name)
    (target/'lambda_ablation_source.json').write_text(json.dumps(dict(source=str(source),
        shared_inputs=required, gains_copied=False, note='Refit gains only; do not resample or reinitialize'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    print('Independent ablation inputs are ready; run fit-hot --anchor-lambda 0 followed by export-hot')

if __name__ == '__main__':
    main()
