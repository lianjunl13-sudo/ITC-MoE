import torch
import torch.nn.functional as F


def effective(core,ue,uo,ui,e):
    return (uo @ torch.einsum('a,aoi->oi',ue[e],core)) @ ui.T


def covariance(x,weight,prior):
    c0=prior.double();c0=(c0+c0.T)*.5
    d=c0.shape[0];prior_scale=c0.trace()/d
    assert torch.isfinite(c0).all() and prior_scale>0,'Invalid prior second moment'
    if x.shape[0]:
        x=x.double();w=weight.double().square()
        total=w.sum().clamp_min(1e-30)
        ch=(x.T*w)@x/total
        neff=float(total.square()/w.square().sum().clamp_min(1e-30))
        beta=neff/(neff+128.)
        
        scale=(ch.trace()/d).clamp_min(prior_scale*1e-8)
        c0=c0*(scale/prior_scale)
        c=beta*ch+(1-beta)*c0
    else:
        neff=0.;beta=0.;scale=prior_scale;c=c0
    c=c+torch.eye(d,device=c.device,dtype=c.dtype)*(1e-4*scale)
    chol,info=torch.linalg.cholesky_ex(c)
    assert not bool(info.any()) and torch.isfinite(chol).all(),'Cholesky factorization failed for the regularized covariance'
    return chol,dict(samples=x.shape[0],effective_samples=neff,beta=beta,scale=float(scale))


def weighted_svd(residual,chol,rank=11,exact_check=False):
    m=residual.float()@chol.float()
    
    u,s,v=torch.svd_lowrank(m,q=min(rank+24,min(m.shape)),niter=3)
    a=u[:,:rank]*s[:rank].sqrt()
    bwhite=s[:rank,None].sqrt()*v[:,:rank].T
    b=torch.linalg.solve_triangular(chol.T,bwhite.T.double(),upper=True).T.float()
    delta=a@b
    err=((residual-delta)@chol.float()).square().sum()
    initial=m.square().sum()
    assert torch.isfinite(a).all() and torch.isfinite(b).all() and err<=initial*1.0001,'Weighted residual did not decrease or contains non-finite values'
    report=dict(weighted_relative_error=float(err/initial.clamp_min(1e-30)),captured_fraction=float(1-err/initial.clamp_min(1e-30)))
    if exact_check:
        _,ss,_=torch.linalg.svd(m,full_matrices=False)
        best=ss[rank:].square().sum()
        ratio=float(err/best.clamp_min(1e-30))
        report['approx_to_exact_objective_ratio']=ratio
        assert ratio<1.01,'Randomized SVD error exceeds the exact optimum by more than 1%; stop for inspection'
    return a,b,report


def grouped(x,ids,matrices):
    out=x.new_empty((x.shape[0],matrices.shape[1]))
    for e in torch.unique(ids).tolist():
        index=torch.nonzero(ids==e).flatten()
        out[index]=F.linear(x[index],matrices[e])
    return out


def lowrank(x,ids,a,b,g):
    chunks=[]
    for start in range(0,len(x),256):
        xx=x[start:start+256];ee=ids[start:start+256]
        z=torch.bmm(b[ee],xx.unsqueeze(-1)).squeeze(-1)*g[ee]
        chunks.append(torch.bmm(a[ee],z.unsqueeze(-1)).squeeze(-1))
    return torch.cat(chunks)


def moe(data,base,adapters=None,gains=None):
    x,experts,weights=data
    ids=experts.reshape(-1)
    pos=torch.arange(len(x),device=x.device)[:,None].expand_as(experts).reshape(-1)
    rx=x[pos]
    def project(value,j):
        out=grouped(value,ids,base[j])
        if adapters is not None:
            a,b=adapters[j]
            out=out+lowrank(value,ids,a,b,gains[j])
        return out
    gate=project(rx,0);up=project(rx,1)
    down=project(F.silu(gate)*up,2)
    return (down*weights.reshape(-1,1)).reshape(len(x),experts.shape[1],x.shape[1]).sum(dim=1)
