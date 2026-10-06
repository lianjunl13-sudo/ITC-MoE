import hashlib
import json
import os
from pathlib import Path
import time
import torch
from decoder import BD3withChatTemplate_decoded_jump_expert_limit_speculative
from opencompass.registry import MODELS


@MODELS.register_module(name='ITCMoETimed')
class ITCMoETimed(BD3withChatTemplate_decoded_jump_expert_limit_speculative):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._counts=dict(total=0,denoise=0,store_kv=0)
        self._sample_index=0
        def count(module,args,kwargs):
            self._counts['total']+=1
            self._counts['store_kv' if kwargs.get('store_kv',False) else 'denoise']+=1
        self._count_hook=self.model.register_forward_pre_hook(count,with_kwargs=True)

    def generate(self,inputs,*args,**kwargs):
        assert len(inputs)==1, 'Per-sample statistics require batch=1'
        if 'ITCMOE_EVAL_SEED' in os.environ:
            seed=int(os.environ['ITCMOE_EVAL_SEED'])+self._sample_index
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        self._sample_index+=1
        before=dict(self._counts)
        torch.cuda.synchronize();started=time.perf_counter()
        outputs=super().generate(inputs,*args,**kwargs)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        row=dict(pid=os.getpid(),model=self.path,n=len(inputs),generation_seconds=seconds,
            forward_total=self._counts['total']-before['total'],
            forward_denoise=self._counts['denoise']-before['denoise'],
            forward_store_kv=self._counts['store_kv']-before['store_kv'],
            input_sha256=hashlib.sha256(json.dumps(inputs,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest(),
            outputs_sha256=hashlib.sha256(json.dumps(outputs,ensure_ascii=False).encode()).hexdigest(),
            empty=not str(outputs[0]).strip())
        path=Path(os.environ['ITCMOE_SAMPLE_TIMING_PATH']);path.parent.mkdir(exist_ok=True,parents=True)
        with path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        self.last_generation_metrics=row
        return outputs
