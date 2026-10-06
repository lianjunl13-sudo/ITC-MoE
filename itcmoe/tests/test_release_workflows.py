"""Check three-budget entry points and independent model views without loading a model."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ReleaseWorkflowTests(unittest.TestCase):
    def test_reuse_calibration_skips_recollection(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/run_pipeline.py'),
            '--config',str(ROOT/'configs/sdar_c30.json'),
            '--original-model','unused','--work-dir','unused-c30',
            '--calibration-jsonl','unused','--mbpp-source-jsonl','unused',
            '--opencompass-root','unused','--data-dir','unused',
            '--reuse-calibration-from','unused-c10','--dry-run'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        stages=[json.loads(line)['stage'] for line in result.stdout.splitlines() if line.startswith('{"stage":')]
        self.assertNotIn('covariance',stages)
        self.assertNotIn('rank-inputs',stages)
        self.assertIn('prepare',stages)
        self.assertIn('rank',stages)

    def test_three_budgets_and_operator_flags(self):
        for budget in (10,20,30):
            for mode in ('on','off'):
                result=subprocess.run([sys.executable,str(ROOT/'run.py'),'evaluate',
                    '--budget',str(budget),'--original-model','unused','--work-dir','unused',
                    '--data-dir','unused','--opencompass-root','unused','--operator',mode,'--dry-run'],
                    capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
                plan=json.loads(result.stdout)
                self.assertEqual(plan['runtime']['operator'],mode)
                self.assertEqual(plan['budget'],budget)

    def test_install_runtime_preserves_source_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';target=Path(d)/'target';source.mkdir()
            (source/'config.json').write_text('{}')
            (source/'model.safetensors.index.json').write_text('{}')
            (source/'part.safetensors').write_bytes(b'unit-test-placeholder')
            result=subprocess.run([sys.executable,str(ROOT/'scripts/install_runtime.py'),
                '--source',str(source),'--output',str(target)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual((source/'config.json').read_text(),'{}')
            self.assertEqual((target/'part.safetensors').read_bytes(),(source/'part.safetensors').read_bytes())
            self.assertTrue(json.loads((target/'config.json').read_text())['itc_operator_enabled'])
            self.assertEqual((target/'modeling_sdar_moe.py').read_bytes(),(ROOT/'src/runtime/modeling_sdar_moe.py').read_bytes())

    def test_lambda_zero_is_explicit_training_argument(self):
        result=subprocess.run([sys.executable,str(ROOT/'run.py'),'fit-hot','--original-model','unused',
            '--work-dir','unused','--anchor-lambda','0','--dry-run'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        command=json.loads(result.stdout)['commands'][0]
        self.assertEqual(command[command.index('--anchor-lambda')+1],'0.0')

if __name__=='__main__':unittest.main()
