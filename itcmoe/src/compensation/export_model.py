import argparse
import math
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['initial','final'],required=True);a=p.parse_args()
    setup();count=0
    index=json.loads((BASE/'model.safetensors.index.json').read_text())
    for li in range(48):
        initial=ROOT/'initial'/f'layer-{li}-residual.safetensors'
        assert initial.exists(),'Incomplete SVD shards'
        with safe_open(initial,framework='pt') as f:tensors={k:f.get_tensor(k) for k in f.keys()}
        if a.stage=='final':
            gain=torch.load(ROOT/'gains'/f'layer-{li}.pt',map_location='cpu',weights_only=False)['gain']
            for j,r in enumerate(ROLES):
                k=prefix(li,r)+'.residual_A'
                tensors[k]=(tensors[k].float()*gain[j,:,None,:]).half().contiguous()
        assert all(torch.isfinite(v).all() for v in tensors.values()),'Exported weights contain non-finite values'
        dst=MODEL/initial.name
        save_file(tensors,dst.with_suffix('.tmp'),metadata={'format':'pt'});dst.with_suffix('.tmp').replace(dst)
        for k,v in tensors.items():
            assert k not in index['weight_map'],'Duplicate index entry'
            index['weight_map'][k]=dst.name;count+=v.numel()
    assert count==EXTRA,'Saved compensation budget does not match'
    basecfg=json.loads((BASE/'config.json').read_text())
    basecfg.update(hot_svd_rank=RANK,hot_svd_method='Real Hot-weighted residual SVD and spectral gains',
                   hot_svd_stage=a.stage,td_moe_ratio=basecfg['td_moe_ratio']-EXTRA/ORIGINAL_EXPERT)
    index['metadata']['total_size']+=count*2
    save(MODEL/'config.json',basecfg);save(MODEL/'model.safetensors.index.json',index)
    save(ROOT/f'export_{a.stage}.json',dict(status='PASS',extra_numel=count,
         final_expert_compression=basecfg['td_moe_ratio'],added_expert_fraction=count/ORIGINAL_EXPERT,
         rank=RANK,stage=a.stage,base_unchanged=True,
         adapter_hashes={p.name:sha(p) for p in MODEL.glob('*residual.safetensors')}))
    status('Model export complete',export_stage=a.stage,final_expert_compression=basecfg['td_moe_ratio'])


if __name__=='__main__':main()
