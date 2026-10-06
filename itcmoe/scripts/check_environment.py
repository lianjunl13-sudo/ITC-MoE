"""Check dependency versions and runtime imports, with optional CPU-only checks."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-only',action='store_true')
    a=parser.parse_args()
    names=['torch','numpy','safetensors']
    if not a.cpu_only:names+=['transformers','accelerate','datasets','mmengine','opencompass','triton','flash-attn']
    report=dict(versions={},errors=[],warnings=[])
    for name in names:
        try:report['versions'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            try:
                module=importlib.import_module(name.replace('-','_'))
                report['versions'][name]=getattr(module,'__version__','unknown')
                report['warnings'].append('Import works but distribution metadata is missing: '+name)
            except Exception as error:
                report['errors'].append(f'Cannot import {name}: {type(error).__name__}: {error}')
    if not a.cpu_only:
        sys.path.insert(0,str(ROOT/'src/evaluation'))
        for name in ('src.runtime.modeling_sdar_moe','decoder','itcmoe_timed','arithmetic'):
            try:
                sys.path.insert(0,str(ROOT))
                importlib.import_module(name)
            except Exception as e:report['errors'].append(f'Cannot import {name}: {type(e).__name__}: {e}')
    report['status']='PASS' if not report['errors'] else 'FAIL'
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['errors']:raise SystemExit(1)

if __name__=='__main__':main()
