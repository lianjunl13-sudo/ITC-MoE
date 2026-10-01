import argparse
import os
import sys
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from parameter_count import parameter_count
from shared_basis import get_projection

CODE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ['ITCMOE_WORKDIR']) / 'base_training'
ROOT.mkdir(parents=True, exist_ok=True)
REF = Path(os.environ['ITCMOE_REFERENCE'])
SOURCE = Path(os.environ['ITCMOE_ORIGINAL_MODEL'])
MODEL = Path(os.environ['ITCMOE_WORKDIR']) / 'base_model'
ROLES = ('gate', 'up', 'down')
SHAPES = ((128,768,2048), (128,768,2048), (128,2048,768))

DELTAS = ((0,0,0), (0,0,0), (0,0,0))
STEPS = ((4,16,32), (4,16,32), (4,32,16))
ORIGINAL_EXPERT = 28991029248
PUBLIC = 1541093376
TARGET_LAYER = ORIGINAL_EXPERT / 48 * .8


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def module(name):
    p=CODE/'tools'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def common_args():
    return SimpleNamespace(original_path=SOURCE,model_path=SOURCE,device='cuda',dtype='float16',
        calib_samples=64,calib_seq_len=256,calib_batch_size=1,max_tokens_per_layer=1024,seed=13037,
        whiten_type='both',output_whiten_mode='inverse',output_whiten_roles=['down'],
        cholesky_eps=.01,rand_oversample=16,rand_iters=1)


def collect():
    rank=module('itcmoe_rank_calibration')
    loader=module('itcmoe_calibration_loader')
    diag=module('itcmoe_diagnostics')
    cfg=json.loads((SOURCE/'config.json').read_text());args=common_args()
    if not (ROOT/'rank_routed.pt').exists():
        data,info=rank.build_gsm_mbpp_routed_inputs(args,cfg,list(range(48)),loader,
            REF/'calibration.jsonl','two_bucket_equal_interleaved',ROOT/'rank_sampling.json')
        torch.save(data,ROOT/'rank_routed.pt.tmp');(ROOT/'rank_routed.pt.tmp').replace(ROOT/'rank_routed.pt')
        del data;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'calibration_cache.json',{'rank_routed.pt':sha(ROOT/'rank_routed.pt')})


def tensor_count(ranks):
    shapes=ranks.new_tensor(SHAPES)
    return (ranks.prod(dim=1)+(ranks*shapes).sum(dim=1)).sum()


def normalized_ranks(theta):
    dims=theta.new_tensor(SHAPES);delta=theta.new_tensor(DELTAS)
    lo=dims*.5;hi=dims-2*theta.new_tensor(STEPS)
    with torch.no_grad():
        left=theta.new_tensor(-30.);right=theta.new_tensor(30.)
        for _ in range(40):
            shift=(left+right)/2
            r=lo+(hi-lo)*torch.sigmoid(theta+shift)
            if tensor_count(r+delta)<TARGET_LAYER:left=shift
            else:right=shift
        shift=(left+right)/2
    s=torch.sigmoid(theta+shift);r=lo+(hi-lo)*s;hot=r+delta
    residual=tensor_count(hot)-TARGET_LAYER
    derivative=((hot.prod(dim=1,keepdim=True)/hot+dims)*(hi-lo)*s*(1-s)).sum().detach()
    
    shift=shift-(residual-residual.detach())/derivative.clamp_min(1e-12)
    return lo+(hi-lo)*torch.sigmoid(theta+shift)


def integer_ranks(rho):
    desired=rho.detach().cpu().tolist()
    ranks=[[max(d//2,min(d-2*s,round(v/s)*s)) for v,d,s in zip(row,shape,step)]
           for row,shape,step in zip(desired,SHAPES,STEPS)]
    def total(rr):return sum(parameter_count(shape,[v+a for v,a in zip(row,delta)])
                             for row,shape,delta in zip(rr,SHAPES,DELTAS))
    def distance(rr):return sum(((v-t)/d)**2 for row,des,shape in zip(rr,desired,SHAPES) for v,t,d in zip(row,des,shape))
    for _ in range(100):
        current=total(ranks);err=abs(current-TARGET_LAYER)
        options=[]
        for j in range(3):
            for a in range(3):
                for sign in (-1,1):
                    rr=[r[:] for r in ranks];rr[j][a]+=sign*STEPS[j][a]
                    if not SHAPES[j][a]//2<=rr[j][a]<=SHAPES[j][a]-2*STEPS[j][a]:continue
                    error=abs(total(rr)-TARGET_LAYER)
                    if error<err:options.append((error,distance(rr),rr))
        if not options:break
        ranks=min(options,key=lambda v:(v[0],v[1]))[-1]
    hot=[[r+d for r,d in zip(row,delta)] for row,delta in zip(ranks,DELTAS)]
    assert abs(total(ranks)/TARGET_LAYER-1)<.006, 'Discretization exceeds the tolerance for the 20% budget'
    for b,h,dims in zip(ranks,hot,SHAPES):assert all(0<x==y<d for x,y,d in zip(b,h,dims))
    return ranks,hot


def grouped(x, experts, effective):
    out=x.new_empty((x.shape[0],effective.shape[1]))
    for e in torch.unique(experts):
        mask=experts==e
        out[mask]=x[mask]@effective[int(e)].T
    return out


def factor_forward(x,experts,factors,masks=None):
    core,ue,uo,ui=factors
    z=x.to(ui.dtype)@ui
    if masks is not None:
        ue=ue*masks[0].to(ue.dtype)
        z=z*masks[2].to(z.dtype)
    effective=torch.einsum('ea,aoi->eoi',ue,core)
    mid=grouped(z,experts,effective)
    if masks is not None:mid=mid*masks[1].to(mid.dtype)
    return (mid@uo.T).float()


def moe_forward(data,linears):
    x,selected,weights=data
    experts=selected.reshape(-1)
    pos=torch.arange(x.shape[0],device=x.device)[:,None].expand_as(selected).reshape(-1)
    rx=x[pos]
    gate=linears[0](rx,experts);up=linears[1](rx,experts)
    down=linears[2](F.silu(gate)*up,experts)
    out=torch.zeros_like(x,dtype=torch.float32)
    out.index_add_(0,pos,down*weights.reshape(-1,1))
    return out,gate,up,down


def gpu_data(row,idx=None):
    vals=[row[k].cuda() for k in ('x','experts','weights')]
    vals[0]=vals[0].float();vals[2]=vals[2].float()
    if idx is not None:vals=[v[idx] for v in vals]
    return vals


def refs(data,original,rank):
    with torch.no_grad():
        return moe_forward(data,[lambda x,e,w=w:rank.routed_linear_dense(x,e,w) for w in original])


def train_rank(li,data,original,factors,metric,rank,steps):
    ref=refs(data,original,rank)
    confidence=data[2].max(dim=-1).values
    token_weight=1+2*(confidence-confidence.min())/(confidence.max()-confidence.min()).clamp_min(1e-12)
    theta=torch.nn.Parameter(torch.zeros(3,3,device='cuda'))
    optimizer=torch.optim.Adam([theta],lr=.02)
    logs=[];best=(float('inf'),None,-1);loss_scale=65536.
    for step in range(steps):
        rho=normalized_ranks(theta)
        temp=96*(8/96)**(step/max(1,steps-1))
        masks=[[torch.sigmoid((rho[j,a]-torch.arange(1,factors[j][0].shape[a]+1,device='cuda'))/(temp*SHAPES[j][a]/2048))
                for a in range(3)] for j in range(3)]
        out=moe_forward(data,[lambda x,e,j=j:factor_forward(x,e,factors[j],masks[j]) for j in range(3)])[0]
        loss=rank.quadratic_mse(ref[0],out,metric,token_weight)
        if not torch.isfinite(loss):raise RuntimeError(f'Non-finite rank loss at layer {li}')
        value=float(loss.detach())
        if value<best[0]:best=(value,rho.detach().clone(),step)
        optimizer.zero_grad(set_to_none=True)
        
        for retry in range(16):
            gt,gr=torch.autograd.grad(loss*loss_scale,(theta,rho),retain_graph=True)
            if torch.isfinite(gt).all() and torch.isfinite(gr).all():break
            loss_scale/=2
        else:raise RuntimeError(f'Scaled rank backward remains non-finite at layer {li}')
        theta.grad=gt/loss_scale
        if step==0:
            initial_grad=theta.grad.detach().cpu().tolist()
            raw_initial_grad=(gr/loss_scale).detach().cpu().tolist()
            assert torch.count_nonzero(gr)==9, 'Not all nine projection/mode gradients are active at the first step'
        optimizer.step()
        row=dict(step=step,loss=value,loss_scale=loss_scale,cold_rho=rho.detach().cpu().tolist(),hot_params=float(tensor_count(rho+rho.new_tensor(DELTAS)).detach()))
        logs.append(row)
        if step%15==0 or step==steps-1:print('RANK',li,json.dumps(row),flush=True)
        del out,loss,masks,gt,gr
    cold,hot=integer_ranks(best[1])
    return cold,hot,dict(best_loss=best[0],best_step=best[2],initial_gradient=initial_grad,
                         initial_unconstrained_rho_gradient=raw_initial_grad,history=logs,
                         continuous_cold=best[1].cpu().tolist())




def train(layers,rank_steps=90,hot_steps=0):
    assert hot_steps==0 and all(v==0 for row in DELTAS for v in row), 'Compensation must be disabled during base-model training'
    rank=module('itcmoe_rank_calibration');td=module('itcmoe_decompose')
    args=common_args();cov=torch.load(REF/'covariance_reference.pt',map_location='cpu',weights_only=False)
    routed=torch.load(ROOT/'rank_routed.pt',map_location='cpu',weights_only=False)
    library=Path(os.environ.get('ITCMOE_FACTOR_LIBRARY', str(Path(os.environ['ITCMOE_WORKDIR'])/'shared_factors')))
    covariance_sha=sha(REF/'covariance_reference.pt')
    decompose_sha=sha(CODE/'tools/itcmoe_decompose.py')
    plan_path=ROOT/'rank_plan.json'
    plan=json.loads(plan_path.read_text()) if plan_path.exists() else dict(cold_ranks={},hot_ranks={},layers={})
    MODEL.mkdir(parents=True,exist_ok=True)
    for li in layers:
        output=MODEL/f'layer-{li}-tdmoe.safetensors'
        if str(li) in plan['layers']:
            assert output.exists() and sha(output)==plan['layers'][str(li)]['weight_sha256'], 'Resume-layer weights do not match the recorded hash'
            continue
        started=time.time();print('Training layer',li,flush=True)
        save(ROOT/'training_status.json',dict(layer=li,stage='Loading original layer and maximum-rank basis',time=time.time(),pid=os.getpid()))
        with safe_open(SOURCE/f'layer-{li}-ep-0-of-1.safetensors',framework='pt',device='cpu') as f:
            tensors={k:f.get_tensor(k) for k in f.keys()}
        original=[torch.stack([tensors[rank.expert_key(li,e,role+'_proj')] for e in range(128)]).cuda().float() for role in ROLES]
        identity=dict(source_sha256=sha(SOURCE/f'layer-{li}-ep-0-of-1.safetensors'), covariance_sha256=covariance_sha, decompose_sha256=decompose_sha)
        factors=[]
        for j,role in enumerate(ROLES):
            incov=cov['input']['hidden' if role!='down' else 'intermediate'][li]
            outcov=cov['output']['down'][li] if role=='down' else None
            maximum=tuple(d-2*s for d,s in zip(SHAPES[j],STEPS[j]))
            torch.manual_seed(13037+li*31+j)
            print('Decomposing',li,role,maximum,flush=True)
            fs=get_projection(library,li,role,maximum,original[j].cpu(),incov,outcov,args,td.decompose_weight_tensor,identity)
            factors.append([v.cuda() for v in fs]);del fs;gc.collect();torch.cuda.empty_cache()
        metric=cov['output']['down'][li].cuda().float();metric=(metric+metric.T)*.5
        save(ROOT/'training_status.json',dict(layer=li,stage='Learning three-mode ranks',time=time.time(),pid=os.getpid()))
        cold,hot,rlog=train_rank(li,gpu_data(routed[li]),original,factors,metric,rank,rank_steps)
        save(ROOT/'layers'/f'{li:02d}_ranks.json',dict(cold=cold,hot=hot,training=rlog))
        
        base=[]
        for f,b in zip(factors,cold):
            base.append([f[0][:b[0],:b[1],:b[2]].contiguous(),*[u[:,:r].contiguous() for u,r in zip(f[1:],b)]])
        del factors,metric;gc.collect();torch.cuda.empty_cache()
        save(ROOT/'training_status.json',dict(layer=li,stage='Exporting base model without compensation',ranks=cold,time=time.time(),pid=os.getpid()))
        exported=[{k:v.detach().cpu().half().contiguous() for k,v in zip(('core','u_exp','u_out','u_in'),fs)} for fs in base]
        hlog=dict(enabled=False,steps=0,extra_parameters=0,reason='Residual compensation is disabled during base-model training')
        new={k:v.contiguous() for k,v in tensors.items() if not td.is_expert_weight(k)}
        for role,fs in zip(ROLES,exported):
            for k,v in fs.items():assert torch.isfinite(v).all(), f'Non-finite export value at layer {li}, {role}/{k}'
            td.add_tucker_tensors(new,li,role,fs['core'],fs['u_exp'],fs['u_out'],fs['u_in'])
        tmp=output.with_suffix('.safetensors.tmp');save_file(new,tmp,metadata={'format':'pt'});tmp.replace(output)
        plan['cold_ranks'][str(li)]=dict(zip(ROLES,cold));plan['hot_ranks'][str(li)]=dict(zip(ROLES,hot))
        plan['layers'][str(li)]=dict(rank=rlog,compensation=hlog,seconds=time.time()-started,
            expert_params=sum(parameter_count(s,r) for s,r in zip(SHAPES,hot)),weight_sha256=sha(output))
        save(plan_path,plan)
        print('Completed layer',li,'elapsed seconds',time.time()-started,'Cold',cold,'Hot',hot,flush=True)
        del original,tensors,base,exported,new;gc.collect();torch.cuda.empty_cache()
    if set(plan['layers'])==set(map(str,range(48))):pack(plan)


def pack(plan):
    td=module('itcmoe_decompose');td.copy_model_side_files(SOURCE,MODEL)
    shutil.copy2(CODE/'runtime/modeling_sdar_moe.py',MODEL/'modeling_sdar_moe.py')
    shutil.copy2(CODE/'runtime/configuration_sdar_moe.py',MODEL/'configuration_sdar_moe.py')
    config=json.loads((SOURCE/'config.json').read_text())
    expert=sum(v['expert_params'] for v in plan['layers'].values())
    config.update(itc_runtime_version='20260925-shared-v1',itc_operator_enabled=True,itc_candidate_enabled=True,itc_candidate_size=48,itc_hot_enabled=True,td_moe_enabled=True,td_moe_kernel='triton',td_moe_operator='hybrid_down_effective',
        td_moe_chunk_size=16384,td_moe_effective_cache_max_gb=24.,td_moe_effective_cache_reserve_gb=4.,
        td_moe_whiten_type='both',td_moe_output_whiten_mode='inverse',td_moe_output_whiten_roles=['down'],
        td_moe_rank_policy='three_mode_normalized_soft_rank',td_moe_ratio=1-expert/ORIGINAL_EXPERT,
        td_moe_ranks_by_layer_role=plan['hot_ranks'],td_moe_cold_ranks_by_layer_role=plan['cold_ranks'],
        td_moe_ranks=plan['hot_ranks']['0'],td_moe_cold_ranks=plan['cold_ranks']['0'],
        td_moe_hotcold_runtime_rank_enabled=False,td_moe_compensation_enabled=False,
        td_moe_trunc_compensation=dict(enabled=False,method='none',
            hot_rank_delta=dict(zip(ROLES,DELTAS)),steps_hot=0,steps_cold=0))
    index={};numel=0
    for p in sorted(MODEL.glob('*.safetensors')):
        with safe_open(p,framework='pt',device='cpu') as f:
            for key in f.keys():
                assert key not in index, 'Duplicate weight key: '+key
                index[key]=p.name;numel+=math.prod(f.get_slice(key).get_shape())
    assert numel==expert+PUBLIC, f'Total parameter count mismatch: {numel} vs {expert+PUBLIC}'
    assert .195<=1-expert/ORIGINAL_EXPERT<=.205, 'Final expert compression is outside the 20% tolerance'
    save(MODEL/'config.json',config)
    save(MODEL/'model.safetensors.index.json',dict(metadata=dict(total_size=numel*2),weight_map=index))
    save(ROOT/'final_parameter_audit.json',dict(status='PASS',expert_numel=expert,total_numel=numel,
        expert_compression=1-expert/ORIGINAL_EXPERT,total_compression=1-numel/(ORIGINAL_EXPERT+PUBLIC),
        groups=144,all_three_modes_strictly_reduced=True,compensation_enabled=False,compensation_parameters=0,model=str(MODEL)))


if __name__=='__main__':
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['collect','train'])
    p.add_argument('--layers',type=int,nargs='*',default=list(range(48)))
    p.add_argument('--rank-steps',type=int,default=90);p.add_argument('--hot-steps',type=int,default=0)
    a=p.parse_args()
    if a.stage=='collect':collect()
    else:train(a.layers,a.rank_steps,a.hot_steps)
