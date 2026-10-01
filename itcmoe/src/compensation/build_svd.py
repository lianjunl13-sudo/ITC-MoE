import argparse
import gc
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from common import *
from residual_math import covariance,weighted_svd,effective


def load_data(li,round_id,split):
    rows=[]
    for p in sorted((ROOT/f'round{round_id}').glob('prompt-*.pt')):
        data=torch.load(p,map_location='cpu',weights_only=False)
        if data['source']['split']==split:rows.append(data['layers'][li])
    assert rows, 'No real Hot cache found for the requested split'
    return [torch.cat([row[k] for row in rows]).cuda().to(dtype) for k,dtype in
            (('x',torch.float32),('experts',torch.long),('weights',torch.float32))]


def layer_weights(li):
    with safe_open(BASE/f'layer-{li}-tdmoe.safetensors',framework='pt',device='cpu') as f:
        factors=[[f.get_tensor(prefix(li,r)+'.'+k).cuda().float() for k in ('core','u_exp','u_out','u_in')] for r in ROLES]
    with safe_open(SOURCE/f'layer-{li}-ep-0-of-1.safetensors',framework='pt',device='cpu') as f:
        originals=[torch.stack([f.get_tensor(f'model.layers.{li}.mlp.experts.{e}.{r}_proj.weight') for e in range(128)]).cuda().float() for r in ROLES]
    return factors,originals


def main():
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,nargs='+',default=list(range(48)));a=p.parse_args()
    setup();out=ROOT/'initial';out.mkdir(exist_ok=True)
    assert (ROOT/'round1/complete.json').exists(),'Full training-source calibration is incomplete'
    cov=torch.load(REF/'covariance_reference.pt',map_location='cpu',weights_only=False)
    for li in a.layers:
        dst=out/f'layer-{li}-residual.safetensors'
        if dst.exists():
            assert sha(dst)==json.loads((out/f'layer-{li}.json').read_text())['sha256'];continue
        started=time.time();status('Building Hot-weighted SVD',layer=li)
        x,ids,weights=load_data(li,1,'train');factors,originals=layer_weights(li)
        priors=[cov['input']['hidden'][li].cuda(),cov['input']['intermediate'][li].cuda()]
        tensors={};reports=[]
        for e in range(128):
            ix,slot=torch.where(ids==e);xe=x[ix];we=weights[ix,slot]
            hidden_chol,hidden_info=covariance(xe,we,priors[0])
            current=[]
            for j,role in enumerate(ROLES):
                torch.manual_seed(SEED+li*512+e*3+j)
                wb=effective(*factors[j],e)
                if role=='down':
                    
                    xi=F.silu(F.linear(xe,current[0]))*F.linear(xe,current[1])
                    chol,info=covariance(xi,we,priors[1])
                else:chol,info=hidden_chol,hidden_info
                aa,bb,rep=weighted_svd(originals[j][e]-wb,chol,RANK,exact_check=(e==0 and li in (0,24,47)))
                current.append(wb+aa@bb)
                for name,value in (('residual_A',aa),('residual_B',bb)):
                    key=prefix(li,role)+'.'+name
                    tensors.setdefault(key,[]).append(value.cpu().half())
                reports.append(dict(expert=e,role=role,**info,**rep))
            if e%16==0:status('Building Hot-weighted SVD',layer=li,expert=e,elapsed=time.time()-started)
        tensors={k:torch.stack(v) for k,v in tensors.items()}
        assert sum(v.numel() for v in tensors.values())==EXTRA//48,'Compensation parameter count does not match'
        assert all(torch.isfinite(v).all() for v in tensors.values()),'Exported half-precision factors contain non-finite values'
        save_file(tensors,dst.with_suffix('.tmp'),metadata={'format':'pt'});dst.with_suffix('.tmp').replace(dst)
        save(out/f'layer-{li}.json',dict(layer=li,seconds=time.time()-started,sha256=sha(dst),projections=reports,
                                      calibration_split='training sources only',numel=EXTRA//48))
        del x,ids,weights,factors,originals,priors,tensors,current;gc.collect();torch.cuda.empty_cache()
    status('SVD construction complete',layers=a.layers)


if __name__=='__main__':main()
