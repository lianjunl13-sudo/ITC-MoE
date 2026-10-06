"""Check runtime projection reuse, mixed states, Hot residuals and the PyTorch control."""
import ast
from collections import OrderedDict
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional, Tuple
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/runtime'))
from options import configure_runtime

def load_cpu_runtime():
    path=ROOT/'src/runtime/modeling_sdar_moe.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    wanted={'_sdar_get_tucker_rank','_sdar_get_cold_tucker_rank','SDARMoeTuckerLinear',
        'SDARMoeTuckerGateUp','SDARMoeSparseMoeBlock'}
    start=next(n.lineno for n in tree.body if isinstance(n,ast.Assign) and
               any(isinstance(t,ast.Name) and t.id=='_hot_svd_old_init' for t in n.targets))
    nodes=[n for n in tree.body if getattr(n,'name','') in wanted or n.lineno>=start]
    m=ModuleType('itc_cpu_test_runtime')
    m.__dict__.update(torch=torch,nn=torch.nn,F=torch.nn.functional,Optional=Optional,Tuple=Tuple,
        ACT2FN={'silu':torch.nn.functional.silu},_SDAR_EFFECTIVE_CORE_CACHE=OrderedDict(),
        _SDAR_EFFECTIVE_CORE_CACHE_BYTES=0,
        _sdar_triton_grouped_effective_tucker_mid=lambda *a:None,
        _sdar_triton_routed_tucker_mid=lambda *a:None)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),m.__dict__)
    return m

class OptimizedRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17037)
        torch.set_num_threads(2)
        self.runtime=load_cpu_runtime()

    def block(self,rank):
        cfg=SimpleNamespace(num_experts=5,num_experts_per_tok=3,norm_topk_prob=True,
            td_moe_enabled=True,hidden_size=12,moe_intermediate_size=10,hidden_act='silu',
            td_moe_ranks={'gate':[4,7,8],'up':[4,7,8],'down':[4,8,7]},
            td_moe_cold_ranks={'gate':[3,5,6],'up':[3,5,6],'down':[3,6,5]},
            td_moe_kernel='torch',td_moe_operator='hybrid_down_effective',td_moe_chunk_size=7,
            td_moe_effective_cache_max_gb=0,td_moe_compensation_enabled=True,
            hot_svd_rank=rank,itc_candidate_size=4)
        b=self.runtime.SDARMoeSparseMoeBlock(cfg,0).double().eval()
        with torch.no_grad():
            for p in b.parameters():p.normal_(0,.25)
            for name,value in b.named_buffers():
                if 'scale' in name:value.uniform_(.8,1.2)
        return b

    def test_numeric_equivalence_and_projection_reuse(self):
        for rank in (0,3):
            block=self.block(rank)
            x=torch.randn(11,12,dtype=torch.float64)
            experts=torch.rand(11,5).topk(3,dim=-1).indices
            weights=torch.rand(11,3,dtype=torch.float64).softmax(-1)
            for mask in (None,torch.ones(11,dtype=torch.bool),torch.arange(11)%2==0):
                with self.subTest(rank=rank,mask=str(mask)),torch.no_grad():
                    configure_runtime(block,operator=False,candidate_size=4)
                    old=block._dispatch_to_tucker_experts(x,experts,weights,mask)
                    configure_runtime(block,operator=True,candidate_size=4)
                    counts=dict(gate=0,up=0,down=0)
                    old_input,old_output=self.runtime._input,self.runtime._output
                    def counted_input(p,z,state):
                        if p.role in ('gate','up'):counts[p.role]+=len(z)
                        return old_input(p,z,state)
                    def counted_output(p,z,state,dtype):
                        if p.role=='down':counts['down']+=len(z)
                        return old_output(p,z,state,dtype)
                    self.runtime._input,self.runtime._output=counted_input,counted_output
                    try:new=block._dispatch_to_tucker_experts(x,experts,weights,mask)
                    finally:self.runtime._input,self.runtime._output=old_input,old_output
                    self.assertEqual(counts,dict(gate=11,up=11,down=11))
                    self.assertLess(float((new-old).norm()/old.norm().clamp_min(1e-12)),1e-6)

    def test_candidate_off_preserves_cold_state(self):
        b=self.block(3)
        configure_runtime(b,operator=True,candidate=False,candidate_size=4)
        seen=[]
        def capture(token_hidden_states,selected_experts,routing_weights,cold_token_mask=None):
            seen.append(cold_token_mask.clone())
            return torch.zeros_like(token_hidden_states)
        b._dispatch_to_tucker_experts=capture
        x=torch.randn(1,4,12,dtype=torch.float64)
        cold=torch.tensor([[True,False,True,False]])
        b(x,torch.zeros_like(x),torch.zeros(1,4,dtype=torch.bool),cold)
        self.assertTrue(torch.equal(seen[0],cold.flatten()))

    def test_default_and_independent_switches(self):
        b=self.block(3)
        self.assertTrue(b.shared_projection_enabled)
        configure_runtime(b,operator=False,hot=False,candidate_size=4)
        self.assertFalse(b.shared_projection_enabled)
        self.assertTrue(b.candidate_enabled)
        for p in (b.gate_up_proj.gate_proj,b.gate_up_proj.up_proj,b.down_proj):
            self.assertEqual((p.operator,p.kernel),('plain','torch'))
            self.assertTrue(p.hot_svd_disabled)
        empty=torch.empty(0,12,dtype=torch.float64)
        configure_runtime(b,candidate_size=4)
        self.assertEqual(b._dispatch_to_tucker_experts(empty,torch.empty(0,3,dtype=torch.long),torch.empty(0,3)).shape,empty.shape)

if __name__=='__main__':unittest.main()
