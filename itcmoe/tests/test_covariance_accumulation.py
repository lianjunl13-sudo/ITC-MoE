import ast
from pathlib import Path
import unittest
import torch

path = Path(__file__).resolve().parents[1]/"src/tools/collect_covariance.py"
tree = ast.parse(path.read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "accumulate_if_finite")
namespace = {"torch": torch}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
accumulate = namespace["accumulate_if_finite"]


class AccumulationTests(unittest.TestCase):
    def test_bad_attempt_does_not_change_any_statistics(self):
        totals = [torch.eye(2), torch.eye(2)*2]
        counts = [3, 5]
        valid = accumulate(totals, [torch.ones(2, 2), torch.full((2, 2), float("nan"))], counts, [7, 7])
        self.assertFalse(valid)
        torch.testing.assert_close(totals[0], torch.eye(2))
        torch.testing.assert_close(totals[1], torch.eye(2)*2)
        self.assertEqual(counts, [3, 5])
        self.assertTrue(accumulate(totals, [torch.ones(2, 2), torch.eye(2)], counts, [7, 7]))
        self.assertEqual(counts, [10, 12])
        torch.testing.assert_close(totals[0], torch.eye(2)+torch.ones(2, 2))

    def test_backoff_recovers_fp16_overflow_without_changing_gradient_formula(self):
        scale = 1024.
        totals = [torch.zeros(1, 1)]
        counts = [0]
        attempts = 0
        while True:
            x = torch.tensor([1.], dtype=torch.float16, requires_grad=True)
            (x.float().sum()*100*scale).backward()
            g = x.grad.float()/scale
            attempts += 1
            if accumulate(totals, [g[:, None]@g[None, :]], counts, [1]):
                break
            scale /= 2
        self.assertGreater(attempts, 1)
        self.assertEqual(counts, [1])
        torch.testing.assert_close(totals[0], torch.tensor([[10000.]]))
