"""Alternate complete operator-off and operator-on timing on the same model and GPU."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True, help='JSON array or JSONL; each item must contain prompt or origin_prompt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--max-tokens', type=int, default=256)
    p.add_argument('--seed', type=int, default=16037)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.limit <= 0 or a.max_tokens <= 0 or a.max_tokens % 32:
        raise ValueError('The sample count must be positive and generation length must be a positive multiple of 32')
    plan = dict(model=str(a.model), limit=a.limit, max_tokens=a.max_tokens, seed=a.seed,
                off='plain + torch + no shared projection reuse',
                on='hybrid_down_effective + triton + effective-core cache + shared projection reuse',
                hot=True, candidate_size=48, warmup_each=1, order='alternate order by sample')
    if a.dry_run:
        print(json.dumps(plan, ensure_ascii=False))
        return
    if a.output.exists():
        raise ValueError('The result directory already exists; refusing to overwrite')
    text = a.samples.read_text(encoding='utf-8-sig')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(s) for s in text.splitlines() if s.strip()]
    rows = rows[:a.limit]
    prompts = [r.get('origin_prompt', r.get('prompt')) for r in rows]
    if len(rows) != a.limit or any(p is None for p in prompts):
        raise ValueError('Insufficient samples or missing prompt field')
    a.output.mkdir(parents=True)
    os.environ.update(ITCMOE_OPERATOR='on', ITCMOE_HOT='on', ITCMOE_CANDIDATE='on', ITCMOE_CANDIDATE_SIZE='48')
    os.environ.pop('ITCMOE_EVAL_SEED',None)
    sys.path[:0] = [str(ROOT/'src/evaluation'), str(ROOT/'src/runtime')]
    import torch
    from itcmoe_timed import ITCMoETimed
    from options import configure_runtime
    if not torch.cuda.is_available():
        raise RuntimeError('The timing experiment requires a CUDA GPU')
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    settings=dict(mask_id=151669,gen_length=a.max_tokens,block_length=32,denoising_steps=32,
                  temperature=1.,top_k=1,top_p=1.,cfg_scale=0.,remasking='low_confidence',threshold=.95,speculative=True)
    wrapper=ITCMoETimed(path=str(a.model),model_kwargs=dict(torch_dtype=torch.float16,trust_remote_code=True),generation_kwargs=settings)
    def generate(index, enabled, warmup=False):
        configure_runtime(wrapper.model, operator=enabled)
        torch.manual_seed(a.seed+index)
        torch.cuda.manual_seed_all(a.seed+index)
        label='on' if enabled else 'off'
        os.environ['ITCMOE_SAMPLE_TIMING_PATH']=str(a.output/(('warmup_' if warmup else 'timing_')+label+'.jsonl'))
        before=dict(wrapper._counts)
        torch.cuda.synchronize()
        start=time.perf_counter()
        output=wrapper.generate([prompts[index]],max_out_len=a.max_tokens)
        torch.cuda.synchronize()
        result=dict(index=index,mode=label,seconds=wrapper.last_generation_metrics['generation_seconds'],output=output[0],
                    forward_total=wrapper._counts['total']-before['total'])
        return result
    results=[]
    for mode in (False,True):
        generate(0,mode,True)
    for i in range(len(rows)):
        for mode in ((False,True) if i%2==0 else (True,False)):
            result=generate(i,mode)
            results.append(result)
            with (a.output/'predictions.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(result,ensure_ascii=False)+'\n')
    totals={m:sum(r['seconds'] for r in results if r['mode']==m) for m in ('off','on')}
    report=dict(protocol=plan,totals=totals,speedup=totals['off']/totals['on'],results=results,
                note='Small-sample timing; operation order may alter generation and forward counts, and does not establish full-benchmark quality equivalence')
    (a.output/'complete.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=totals,speedup=report['speedup']),ensure_ascii=False))

if __name__=='__main__':
    main()
