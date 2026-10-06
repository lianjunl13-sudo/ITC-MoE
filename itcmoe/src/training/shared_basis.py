"""Persist a shared whitened ordered factor library for all three rank budgets."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file
from factor_store import load_projection, NAMES

MAXIMA = {'gate': (124, 752, 2016), 'up': (124, 752, 2016), 'down': (124, 2016, 752)}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def get_projection(library, layer, role, maximum, weight, incov, outcov, args, decompose, identity):
    library = Path(library)
    library.mkdir(parents=True, exist_ok=True)
    path = library/f'layer-{layer:02d}-{role}.safetensors'
    record = path.with_suffix('.json')
    if path.exists():
        if not record.exists():
            raise RuntimeError('Existing factors have no checksum record: '+str(path))
        info = json.loads(record.read_text(encoding='utf-8'))
        if sha(path) != info['sha256']:
            raise RuntimeError('Shared-factor hash mismatch: '+str(path))
        if info.get('identity') != identity:
            raise RuntimeError('The shared library uses a different teacher, covariance or decomposition source')
    else:
        j = ('gate', 'up', 'down').index(role)
        torch.manual_seed(13037 + layer*31 + j)
        values = decompose(weight, MAXIMA[role], incov, outcov, args)
        if any(not torch.isfinite(v).all() for v in values):
            raise RuntimeError('Shared decomposition contains non-finite values')
        tmp = path.with_suffix('.partial')
        save_file(dict(zip(NAMES, [v.cpu().contiguous() for v in values])), tmp, metadata={'format':'pt'})
        tmp.replace(path)
        record.write_text(json.dumps(dict(sha256=sha(path), identity=identity, maximum=MAXIMA[role],
            seed=13037+layer*31+j), ensure_ascii=False, indent=2), encoding='utf-8')
    return load_projection(library, layer, role, maximum)
