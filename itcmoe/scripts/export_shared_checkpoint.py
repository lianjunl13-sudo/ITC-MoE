"""Export learned shared factors, ranks and Hot gains as a standalone HF checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def check(path,record):
    if not record.is_file() or sha(path)!=json.loads(record.read_text(encoding='utf-8'))['sha256']:
        raise ValueError('Shard hash verification failed: '+str(path))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-model',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--rank-view',type=Path,required=True)
    p.add_argument('--hot-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('The output directory already exists; refusing to overwrite')
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path.insert(0,str(ROOT/'src/training'))
    from factor_store import load_projection
    view=json.loads(a.rank_view.read_text(encoding='utf-8'))
    ranks=view.get('ranks',view.get('hot_ranks',view.get('ranks_by_layer_role',view)))
    if set(ranks)!=set(map(str,range(48))):raise ValueError('The rank view must contain all 48 layers')
    roles=('gate','up','down')
    for li in range(48):
        for role in roles:
            path=a.library/f'layer-{li:02d}-{role}.safetensors'
            check(path,path.with_suffix('.json'))
        if a.hot_root:
            check(a.hot_root/'initial'/f'layer-{li}-residual.safetensors',a.hot_root/'initial'/f'layer-{li}.json')
            check(a.hot_root/'gains'/f'layer-{li}.pt',a.hot_root/'gains'/f'layer-{li}.json')
    a.output.mkdir(parents=True)
    for source in a.original_model.iterdir():
        if not source.is_file() or source.name=='model.safetensors.index.json':continue
        if source.suffix=='.safetensors':
            if source.name!='others.safetensors':continue
            try:os.link(source,a.output/source.name)
            except OSError:shutil.copy2(source,a.output/source.name)
        else:shutil.copy2(source,a.output/source.name)
    total_expert=0
    for li in range(48):
        with safe_open(a.original_model/f'layer-{li}-ep-0-of-1.safetensors',framework='pt') as f:
            tensors={k:f.get_tensor(k) for k in f.keys() if not ('.mlp.experts.' in k and k.endswith('.weight'))}
        for role in roles:
            prefix=f'model.layers.{li}.mlp.'+('down_proj' if role=='down' else f'gate_up_proj.{role}_proj')
            values=load_projection(a.library,li,role,ranks[str(li)][role])
            for key,value in zip(('core','u_exp','u_out','u_in'),values):
                if not torch.isfinite(value).all():raise ValueError('Shared factors contain non-finite values')
                tensors[prefix+'.'+key]=value.half().contiguous()
                total_expert+=value.numel()
        if a.hot_root:
            with safe_open(a.hot_root/'initial'/f'layer-{li}-residual.safetensors',framework='pt') as f:
                adapter={k:f.get_tensor(k) for k in f.keys()}
            gain=torch.load(a.hot_root/'gains'/f'layer-{li}.pt',map_location='cpu',weights_only=False)['gain']
            if tuple(gain.shape)!=(3,128,11):raise ValueError('This release requires Hot rank 11')
            for j,role in enumerate(roles):
                prefix=f'model.layers.{li}.mlp.'+('down_proj' if role=='down' else f'gate_up_proj.{role}_proj')
                key=prefix+'.residual_A'
                adapter[key]=(adapter[key].float()*gain[j,:,None,:]).half().contiguous()
            if any(not torch.isfinite(v).all() for v in adapter.values()):raise ValueError('Hot compensation contains non-finite values')
            tensors.update(adapter)
            total_expert+=sum(v.numel() for v in adapter.values())
        target=a.output/f'layer-{li}-tdmoe.safetensors'
        save_file(tensors,target,metadata={'format':'pt'})
        print(f'Exported layer {li}',flush=True)
    for name in ('modeling_sdar_moe.py','configuration_sdar_moe.py'):
        shutil.copy2(ROOT/'src/runtime'/name,a.output/name)
    cfg=json.loads((a.original_model/'config.json').read_text(encoding='utf-8'))
    cfg.update(td_moe_enabled=True,td_moe_kernel='triton',td_moe_operator='hybrid_down_effective',
        td_moe_chunk_size=16384,td_moe_effective_cache_max_gb=24.,td_moe_effective_cache_reserve_gb=4.,
        td_moe_ranks_by_layer_role=ranks,td_moe_cold_ranks_by_layer_role=ranks,
        td_moe_ranks=ranks['0'],td_moe_cold_ranks=ranks['0'],td_moe_compensation_enabled=False,
        td_moe_hotcold_runtime_rank_enabled=False,td_moe_ratio=1-total_expert/28991029248,
        hot_svd_rank=11 if a.hot_root else 0,itc_runtime_version='20260925-shared-v1',
        itc_operator_enabled=True,itc_candidate_enabled=True,itc_candidate_size=48,itc_hot_enabled=True)
    index={};size=0;numel=0
    for path in a.output.glob('*.safetensors'):
        with safe_open(path,framework='pt') as f:
            for key in f.keys():
                if key in index:raise ValueError('Duplicate weight key: '+key)
                index[key]=path.name
                value=f.get_tensor(key)
                size+=value.numel()*value.element_size();numel+=value.numel()
    if numel!=total_expert+1541093376:raise ValueError('Exported parameter count does not match')
    def dump(name,value):
        (a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    dump('config.json',cfg)
    dump('model.safetensors.index.json',dict(metadata=dict(total_size=size),weight_map=index))
    dump('export_provenance.json',dict(rank_view_sha256=sha(a.rank_view),expert_parameters=total_expert,
        total_parameters=numel,hot_rank=cfg['hot_svd_rank'],runtime_sha256=sha(a.output/'modeling_sdar_moe.py')))

if __name__=='__main__':main()
