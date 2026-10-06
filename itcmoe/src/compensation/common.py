"""Portable paths and fixed hyperparameters for the released Hot branch."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['ITCMOE_WORKDIR']) / 'hot'
ROOT.mkdir(parents=True, exist_ok=True)
BASE = Path(os.environ['ITCMOE_WORKDIR']) / 'base_model'
MODEL = Path(os.environ['ITCMOE_WORKDIR']) / 'model'
REF = Path(os.environ['ITCMOE_REFERENCE'])
SOURCE = Path(os.environ['ITCMOE_ORIGINAL_MODEL'])
PY = sys.executable
ROLES = ('gate', 'up', 'down')
RANK = 11
SEED = 16037
ORIGINAL_EXPERT = 28991029248
EXTRA = 48 * 128 * 3 * RANK * (2048 + 768)

def prefix(li, role):
    sub = 'down_proj' if role == 'down' else f'gate_up_proj.{role}_proj'
    return f'model.layers.{li}.mlp.{sub}'

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    tmp.replace(path)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()

def status(stage, **kwargs):
    row = dict(stage=stage, pid=os.getpid(), time=time.strftime('%F %T'), **kwargs)
    save(ROOT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

def setup():
    import torch
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)

def reclaim_read_cache():
    if not hasattr(os, 'posix_fadvise'):
        return
    for directory in (BASE, MODEL, SOURCE):
        for path in directory.glob('*.safetensors'):
            with path.open('rb') as stream:
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
