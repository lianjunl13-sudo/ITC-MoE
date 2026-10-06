import argparse
import gc
import types
import sys
import numpy as np
import torch
from common import *


class Collector:
    def __init__(self, model):
        self.context={}; self.tables={}; self.calls=0
        self.rng=torch.Generator(device='cpu').manual_seed(SEED+918)
        model.register_forward_pre_hook(self.model_pre, with_kwargs=True)
        for li, layer in enumerate(model.model.layers):
            block=layer.mlp
            block.register_forward_pre_hook(self.block_pre,with_kwargs=True)
            old=block._dispatch_to_tucker_experts
            def wrapped(this,token_hidden_states,selected_experts,routing_weights,cold_token_mask=None,_old=old,_li=li):
                self.capture(this,_li,token_hidden_states,selected_experts,routing_weights,cold_token_mask)
                return _old(token_hidden_states,selected_experts,routing_weights,cold_token_mask)
            block._dispatch_to_tucker_experts=types.MethodType(wrapped,block)

    def model_pre(self,module,args,kw):
        ids=kw.get('input_ids',args[0] if args else None)
        if ids is None:return
        self.calls+=1
        decoded=kw.get('decoded_index')
        
        first=decoded is None
        phase=0 if first else (1 if float((ids==151669).float().mean())>.66 else
                               2 if float((ids==151669).float().mean())>.33 else 3)
        self.context=dict(call=self.calls,phase=phase,ids=ids.detach().cpu().reshape(-1),
                          positions=kw.get('position_ids'),spec=bool(ids.shape[1]==128 and decoded is not None))

    def block_pre(self,module,args,kw):
        x=args[0] if args else kw['hidden_states']
        past=args[1] if len(args)>1 else kw.get('past_hidden_states')
        decoded=args[2] if len(args)>2 else kw.get('decoded_index')
        if past is None or decoded is None:
            module._hot_sample_positions=torch.arange(x.shape[0]*x.shape[1])
        else:
            module._hot_sample_positions=torch.nonzero(~decoded.detach().cpu().reshape(-1)).flatten()

    def capture(self,block,li,x,experts,weights,cold):
        if getattr(self,'disabled',False):return
        n=x.shape[0]
        positions=block._hot_sample_positions
        assert positions.numel()==n, 'Captured positions are not aligned with computed positions'
        hot=torch.ones(n,dtype=torch.bool) if cold is None else ~cold.detach().cpu().bool()
        eligible=torch.nonzero(hot).flatten()
        if not eligible.numel():return
        context=self.context
        
        if context['spec']:
            blocks=context['ids'].reshape(4,32)
            allowed=[]
            for b in range(4):
                if not any(torch.equal(blocks[b],blocks[a]) for a in allowed):allowed.append(b)
            eligible=eligible[torch.tensor([int(positions[i]//32) in allowed for i in eligible])]
        if not eligible.numel():return
        phase=context['phase']; key=(li,phase)
        cap=self.cap//4
        priority=torch.rand(eligible.numel(),generator=self.rng)
        current=self.tables.get(key)
        oldkeys=current['priority'] if current else torch.empty(0)
        joined=torch.cat([oldkeys,priority]); keep=joined.topk(min(cap,joined.numel())).indices
        newidx=keep[keep>=oldkeys.numel()]-oldkeys.numel()
        if not newidx.numel():return
        take=eligible[newidx]; dev=take.to(x.device)
        new=dict(priority=priority[newidx],x=x[dev].detach().cpu().half(),
                 experts=experts[dev].detach().cpu().long(),weights=weights[dev].detach().cpu().float(),
                 meta=torch.stack([torch.full_like(take,context['call']),positions[take],
                      torch.full_like(take,phase),torch.full_like(take,int(context['spec']))],dim=1))
        oldidx=keep[keep<oldkeys.numel()]
        if current:new={k:torch.cat([current[k][oldidx],v]) for k,v in new.items()}
        self.tables[key]=new

    def start(self,split):
        self.tables={};self.calls=0;self.cap=168 if split=='train' else 128

    def finish(self):
        result={}
        for li in range(48):
            rows=[self.tables[(li,p)] for p in range(4) if (li,p) in self.tables]
            assert rows, f'Layer {li} has no real Hot samples'
            result[li]={k:torch.cat([r[k] for r in rows]) for k in ('x','experts','weights','meta')}
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,choices=[1,2],required=True)
    p.add_argument('--limit',type=int,default=32);a=p.parse_args()
    setup();sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'evaluation'))
    from decoder import BD3withChatTemplate_decoded_jump_expert_limit_speculative as Wrapper
    model_path=BASE if a.round==1 else MODEL
    kwargs=dict(mask_id=151669,gen_length=256,block_length=32,denoising_steps=32,
                temperature=1.,top_k=1,top_p=1.,remasking='low_confidence',threshold=.95,speculative=True)
    status('Loading model for real Hot sampling',round=a.round,model=str(model_path))
    wrapper=Wrapper(path=str(model_path),model_kwargs=dict(torch_dtype=torch.float16,trust_remote_code=True),generation_kwargs=kwargs)
    collector=Collector(wrapper.model)
    prompts=json.loads((ROOT/'prompts.json').read_text())
    target=ROOT/f'round{a.round}';target.mkdir(exist_ok=True)
    for i,row in enumerate(prompts[:a.limit]):
        dest=target/f'prompt-{i:02d}.pt'
        if dest.exists():continue
        started=time.time();collector.start(row['split'])
        if i==0:
            collector.disabled=True
            
            torch.manual_seed(SEED+i);np.random.seed(SEED+i)
            wrapper.generate([row['prompt']],max_out_len=256)
            collector.start(row['split'])
            torch.manual_seed(SEED+i);np.random.seed(SEED+i)
            reference_output=wrapper.generate([row['prompt']],max_out_len=256)[0]
            reference_calls=collector.calls
            collector.disabled=False;collector.start(row['split'])
        torch.manual_seed(SEED+i);np.random.seed(SEED+i)
        status('Collecting real Hot states',round=a.round,prompt_index=i,source_id=row['source_id'],split=row['split'])
        output=wrapper.generate([row['prompt']],max_out_len=256)[0]
        if i==0:
            save(target/'collector_invariance_attempt.json',dict(reference_output=reference_output,output=output,
                 reference_calls=reference_calls,calls=collector.calls))
            assert output==reference_output and collector.calls==reference_calls, 'The collector changed same-seed generation or forward counts'
            save(target/'collector_invariance.json',dict(status='PASS',same_output=True,same_calls=True,calls=collector.calls))
        samples=collector.finish()
        torch.save(dict(source=row,layers=samples),dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)
        save(target/f'prompt-{i:02d}.json',dict(source_id=row['source_id'],split=row['split'],output=output,
             calls=collector.calls,seconds=time.time()-started,counts={li:r['x'].shape[0] for li,r in samples.items()},
             strata='Initial/refresh and high/medium/low mask ratios; deduplicate identical speculative blocks; stratified random-priority sampling per prompt and layer'))
        collector.tables={};del samples;gc.collect()
    if a.limit==32:
        assert len(list(target.glob('prompt-*.pt')))==32
        save(target/'complete.json',dict(status='PASS',prompt_manifest_sha256=sha(ROOT/'prompts.json'),
             max_train_tokens_per_layer=4032,max_dev_tokens_per_layer=1024,
             meta_columns=['forward_call','flattened_position','phase','speculative'],
             source_split='24 training and 8 development prompts, split by source ID',generation_kwargs=kwargs))
    status('Real Hot sampling complete',round=a.round,limit=a.limit)


if __name__=='__main__':main()
