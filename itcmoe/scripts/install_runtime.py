"""Create an independent view of an existing HF checkpoint with the optimized runtime."""
import argparse
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)

def install(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError('Output must be a new directory; the original model cannot be overwritten')
    if not (source/'model.safetensors.index.json').is_file():
        raise ValueError('Input must be a complete sharded HF model; export shared-factor views with export_shared_checkpoint.py first')
    output.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            if item.suffix == '.safetensors':
                link_or_copy(item, output/item.name)
            else:
                shutil.copy2(item, output/item.name)
    for name in ('modeling_sdar_moe.py', 'configuration_sdar_moe.py'):
        shutil.copy2(ROOT/'src/runtime'/name, output/name)
    cfg = json.loads((output/'config.json').read_text(encoding='utf-8'))
    cfg.update(itc_runtime_version='20260925-shared-v1', itc_operator_enabled=True,
               itc_hot_enabled=True, itc_candidate_enabled=True, itc_candidate_size=48)
    (output/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    return output

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(install(a.source, a.output))

if __name__ == '__main__':
    main()
