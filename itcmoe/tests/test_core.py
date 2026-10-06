import importlib.util
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/compensation'))
sys.path.insert(0, str(ROOT/'src/training'))
from residual_math import covariance, weighted_svd, effective, moe
from parameter_count import parameter_count

class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(16037)
        torch.set_num_threads(2)

    def test_parameter_budget(self):
        self.assertEqual(48*128*3*11*(2048+768),570949632)
        self.assertEqual(parameter_count((4,6,8),(2,3,4)),2*3*4+4*2+6*3+8*4)

    def test_empty_hot_covariance_uses_prior(self):
        prior=torch.diag(torch.arange(1,9).float())
        chol,info=covariance(torch.empty(0,8),torch.empty(0),prior)
        expected=prior.double()+torch.eye(8,dtype=torch.float64)*1e-4*prior.double().trace()/8
        torch.testing.assert_close(chol@chol.T,expected)
        self.assertEqual(info['beta'],0)

    def test_weighted_svd_objective(self):
        x=torch.randn(50,32)
        chol,_=covariance(x,torch.rand(50),torch.eye(32))
        residual=torch.randn(24,32)
        a,b,report=weighted_svd(residual,chol,11,exact_check=True)
        self.assertEqual(a.shape,(24,11))
        self.assertEqual(b.shape,(11,32))
        self.assertLessEqual(report['weighted_relative_error'],1)
        self.assertLess(report['approx_to_exact_objective_ratio'],1.01)

    def test_tucker_reconstruction(self):
        core=torch.randn(2,3,4)
        ue,uo,ui=torch.randn(5,2),torch.randn(6,3),torch.randn(7,4)
        expected=torch.einsum('a,aoi,po,qi->pq',ue[2],core,uo,ui)
        torch.testing.assert_close(effective(core,ue,uo,ui,2),expected)

    def test_zero_compensation_and_gain_gradients(self):
        n,e,d,h,r=7,4,12,8,3
        data=(torch.randn(n,d),torch.randint(e,(n,2)),torch.full((n,2),.5))
        base=[torch.randn(e,h,d)*.1,torch.randn(e,h,d)*.1,torch.randn(e,d,h)*.1]
        adapters=[(torch.randn(e,w.shape[1],r)*.1,torch.randn(e,r,w.shape[2])*.1) for w in base]
        gain=torch.nn.Parameter(torch.ones(3,e,r))
        torch.testing.assert_close(moe(data,base,adapters,torch.zeros_like(gain)),moe(data,base),rtol=0,atol=0)
        moe(data,base,adapters,gain).square().mean().backward()
        self.assertTrue(torch.isfinite(gain.grad).all())
        self.assertGreater(float(gain.grad.abs().sum()),0)

if __name__=='__main__':
    unittest.main()
