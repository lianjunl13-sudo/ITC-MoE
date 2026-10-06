import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src/tools"))
from covariance_shrinkage import shrink_covariance, shrinkage_metadata, validate_shrinkage_cache


class ShrinkageTests(unittest.TestCase):
    def test_spectrum_and_scale_invariance(self):
        matrix = torch.tensor([[2., 1.], [1., 2.]])
        result = shrink_covariance(matrix, .25)
        torch.testing.assert_close(result, torch.tensor([[1., .125], [.125, 1.]]))
        torch.testing.assert_close(result, shrink_covariance(matrix * 1e-16, .25))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(result).min()), .75)
        self.assertEqual(float(torch.trace(result)), 2.)

    def test_endpoints_and_zero_statistics(self):
        matrix = torch.diag(torch.tensor([0., 4.]))
        torch.testing.assert_close(shrink_covariance(matrix, 0), torch.eye(2))
        torch.testing.assert_close(shrink_covariance(matrix, 1), matrix / 2)
        torch.testing.assert_close(shrink_covariance(torch.zeros(2, 2)), torch.eye(2))

    def test_invalid_parameters_and_cache(self):
        for eta in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                shrink_covariance(torch.eye(2), eta)
        with self.assertRaises(ValueError):
            validate_shrinkage_cache({"metadata": {"shrinkage_alpha": .25}}, .25)
        cov = {"metadata": shrinkage_metadata(.25)}
        validate_shrinkage_cache(cov, .25)
        with self.assertRaises(ValueError):
            validate_shrinkage_cache(cov, .5)
