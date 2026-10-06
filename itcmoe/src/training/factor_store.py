"""Slice shared Tucker factors by rank without exporting another full HF model."""
from pathlib import Path
import torch
from safetensors import safe_open

NAMES=('core','u_exp','u_out','u_in')


def slice_factors(core,factors,ranks):
    ranks=tuple(int(v) for v in ranks)
    if len(ranks)!=3 or any(not 0<r<=d for r,d in zip(ranks,core.shape)):
        raise ValueError('Truncated ranks exceed the shared core dimensions')
    if any(f.shape[1]!=core.shape[i] for i,f in enumerate(factors)):
        raise ValueError('Factor column counts do not match the shared core')
    return [core[:ranks[0],:ranks[1],:ranks[2]].contiguous(),
            *[f[:,:r].contiguous() for f,r in zip(factors,ranks)]]


def load_projection(library,layer,role,ranks,device='cpu'):
    path=Path(library)/f'layer-{layer:02d}-{role}.safetensors'
    # Read only requested prefixes, without loading other layers or reconstructing all dense experts.
    with safe_open(path,framework='pt',device='cpu') as f:
        shape=f.get_slice('core').get_shape();r=tuple(int(x) for x in ranks)
        if len(r)!=3 or any(not 0<a<=b for a,b in zip(r,shape)):raise ValueError('Requested rank is out of bounds')
        values=[f.get_slice('core')[:r[0],:r[1],:r[2]],
                *[f.get_slice(name)[:,:rank] for name,rank in zip(NAMES[1:],r)]]
    return [v.contiguous().to(device) for v in values]


def cpu_test():
    torch.manual_seed(1)
    core=torch.randn(4,5,6);fs=[torch.randn(7,4),torch.randn(8,5),torch.randn(9,6)]
    rs=(2,3,4);sub=slice_factors(core,fs,rs)
    x=torch.einsum('abc,ea,ob,ic->eoi',*sub)
    masked=core.clone();masked[2:]=0;masked[:,3:]=0;masked[:,:,4:]=0
    y=torch.einsum('abc,ea,ob,ic->eoi',masked,*fs)
    if not torch.allclose(x,y,atol=2e-5,rtol=1e-5):raise RuntimeError('Truncation and masking equivalence check failed')
    try:slice_factors(core,fs,(5,3,4))
    except ValueError:pass
    else:raise RuntimeError('Out-of-bounds check failed')
    return dict(status='PASS',max_error=float((x-y).abs().max()))
