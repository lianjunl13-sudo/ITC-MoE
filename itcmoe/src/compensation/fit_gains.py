import argparse
import gc
import torch
from safetensors import safe_open
from common import *
from residual_math import effective,moe
from build_svd import load_data,layer_weights


def sliced(data,idx):return [v[idx] for v in data]


@torch.no_grad()
def predict(data,base,adapter=None,gains=None):
    return torch.cat([moe([v[i:i+128] for v in data],base,adapter,gains) for i in range(0,len(data[0]),128)])


def main():
    p=argparse.ArgumentParser();p.add_argument('--layers',type=int,nargs='+',default=list(range(48)));p.add_argument('--anchor-lambda',type=float,default=.1);a=p.parse_args()
    if not 0 <= a.anchor_lambda < float('inf'):raise ValueError('The anchor coefficient must be finite and nonnegative')
    setup();root=ROOT/'gains';root.mkdir(exist_ok=True)
    metadata=root/'fit_config.json'
    settings=dict(anchor_lambda=a.anchor_lambda,rank=RANK,steps=32,lr=.01,seed=SEED)
    if metadata.exists() and json.loads(metadata.read_text())!=settings:raise RuntimeError('This gain directory uses different hyperparameters; use a separate directory')
    if not metadata.exists() and any(root.glob('layer-*.pt')):raise RuntimeError('Existing gains have no hyperparameter record and cannot be reused automatically')
    save(metadata,settings)
    assert (ROOT/'round2/complete.json').exists(),'Second-round real-state collection is incomplete'
    for li in a.layers:
        dst=root/f'layer-{li}.pt'
        if dst.exists():continue
        started=time.time();status('Fitting spectral gains',layer=li,step=0)
        train=load_data(li,2,'train');dev=load_data(li,2,'dev')
        factors,teacher=layer_weights(li)
        base=[torch.stack([effective(*fs,e) for e in range(128)]) for fs in factors]
        del factors;gc.collect();torch.cuda.empty_cache()
        with safe_open(ROOT/'initial'/f'layer-{li}-residual.safetensors',framework='pt') as f:
            adapters=[tuple(f.get_tensor(prefix(li,r)+'.'+k).cuda().float() for k in ('residual_A','residual_B')) for r in ROLES]
        one=torch.ones((3,128,RANK),device='cuda');zero=torch.zeros_like(one)
        tt=predict(train,teacher);td=predict(dev,teacher)
        anchor=predict(train,base,adapters,one)
        denominator=tt.square().mean().clamp_min(1e-12).detach()
        def evaluate(g):
            pred=predict(dev,base,adapters,g)
            value=float((pred-td).square().mean()/denominator)
            assert math_isfinite(value),'Non-finite development error'
            return value
        candidates=[dict(name='zero compensation',step=-1,dev_nmse=evaluate(zero)),
                    dict(name='SVD initialization',step=0,dev_nmse=evaluate(one))]
        best=min(candidates,key=lambda v:v['dev_nmse']).copy()
        best_gain=(zero if best['step']==-1 else one).clone()
        gain=torch.nn.Parameter(one.clone());opt=torch.optim.Adam([gain],lr=.01,weight_decay=0)
        torch.manual_seed(SEED+li)
        history=[]
        for step in range(1,33):
            idx=torch.randperm(len(train[0]),device='cuda')[:128]
            opt.zero_grad(set_to_none=True)
            pred=moe(sliced(train,idx),base,adapters,gain)
            main_loss=(pred-tt[idx]).square().mean()/denominator
            anchor_loss=(pred-anchor[idx]).square().mean()/denominator
            loss=main_loss+a.anchor_lambda*anchor_loss
            assert torch.isfinite(loss),'Non-finite training loss; stopping and preserving artifacts'
            loss.backward()
            assert torch.isfinite(gain.grad).all(),'Non-finite spectral-gain gradient'
            norm=torch.nn.utils.clip_grad_norm_([gain],.1)
            opt.step()
            with torch.no_grad():gain.clamp_(0,2)
            history.append(dict(step=step,main=float(main_loss),anchor=float(anchor_loss),grad_norm=float(norm)))
            if step%8==0:
                row=dict(name='gain checkpoint',step=step,dev_nmse=evaluate(gain.detach()))
                checkpoint=root/'checkpoints'/f'layer-{li}-step-{step}.pt'
                checkpoint.parent.mkdir(exist_ok=True)
                torch.save(dict(gain=gain.detach().cpu(),metrics=row),checkpoint)
                candidates.append(row)
                if row['dev_nmse']<best['dev_nmse']:best=row.copy();best_gain=gain.detach().clone()
                status('Fitting spectral gains',layer=li,step=step,best=best)
        assert best['dev_nmse']<=candidates[0]['dev_nmse']+1e-10,'Development selection performs worse than zero compensation'
        torch.save(dict(gain=best_gain.cpu(),candidates=candidates,best=best),dst.with_suffix('.tmp'))
        dst.with_suffix('.tmp').replace(dst)
        save(root/f'layer-{li}.json',dict(layer=li,seconds=time.time()-started,train_tokens=len(train[0]),
             dev_tokens=len(dev[0]),candidates=candidates,best=best,history=history,
             trainable_numel=gain.numel(),fixed_denominator=float(denominator),sha256=sha(dst)))
        del train,dev,teacher,base,adapters,gain,opt,tt,td,anchor,pred,loss,best_gain
        gc.collect();torch.cuda.empty_cache()
        memory_file=Path('/sys/fs/cgroup/memory.current')
        if memory_file.exists() and int(memory_file.read_text())>64*2**30:
            from common import reclaim_read_cache as reclaim
            reclaim()
    status('Spectral-gain fitting complete',layers=a.layers)


def math_isfinite(x):
    import math
    return math.isfinite(x)


if __name__=='__main__':main()
