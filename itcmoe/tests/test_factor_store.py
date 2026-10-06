"""Check persisted shared-factor prefixes and bounds without CUDA."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from safetensors.torch import save_file

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/training'))
from factor_store import load_projection,slice_factors

class FactorStoreTests(unittest.TestCase):
    def test_persisted_prefix_matches_masked_full_tensor(self):
        torch.manual_seed(13037)
        core=torch.randn(4,5,6)
        factors=[torch.randn(7,4),torch.randn(8,5),torch.randn(9,6)]
        with tempfile.TemporaryDirectory() as d:
            save_file(dict(zip(('core','u_exp','u_out','u_in'),[core,*factors])),Path(d)/'layer-00-gate.safetensors')
            values=load_projection(d,0,'gate',(2,3,4))
        expected=slice_factors(core,factors,(2,3,4))
        for a,b in zip(values,expected):self.assertTrue(torch.equal(a,b))
        masked=core.clone();masked[2:]=0;masked[:,3:]=0;masked[:,:,4:]=0
        torch.testing.assert_close(torch.einsum('abc,ea,ob,ic->eoi',*values),
            torch.einsum('abc,ea,ob,ic->eoi',masked,*factors),rtol=1e-5,atol=2e-5)

    def test_out_of_bounds_rank_is_rejected(self):
        with self.assertRaises(ValueError):
            slice_factors(torch.zeros(2,3,4),[torch.zeros(5,2),torch.zeros(6,3),torch.zeros(7,4)],(3,2,2))

if __name__=='__main__':unittest.main()
