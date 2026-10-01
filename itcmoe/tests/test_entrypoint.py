import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]

class EntryTests(unittest.TestCase):
    def test_all_stages_dry_run_without_model_loading(self):
        for stage in ('prepare','covariance','rank-inputs','rank','prepare-hot','hot','evaluate'):
            with self.subTest(stage=stage):
                command=[sys.executable,str(ROOT/'run.py'),stage,'--original-model','unused-model',
                         '--work-dir','unused-work','--data-dir','unused-data',
                         '--opencompass-root','unused-framework','--dry-run']
                result=subprocess.run(command,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(json.loads(result.stdout)['stage'],stage)

    def test_calibration_selection_is_train_only(self):
        rows=json.loads((ROOT/'configs/calibration_selection.json').read_text())
        self.assertEqual(len(rows),256)
        self.assertEqual(len({r['source_id'] for r in rows}),256)
        self.assertTrue(all(r['split']=='train' for r in rows))
        self.assertEqual(sum(r['bucket']=='gsm8k_train' for r in rows),128)
        self.assertEqual(sum(r['bucket']=='mbpp_train' for r in rows),128)
        self.assertTrue(all(len(r['text_sha256'])==64 for r in rows))

    def test_partial_hot_export_is_rejected(self):
        result=subprocess.run([sys.executable,str(ROOT/'run.py'),'hot','--original-model','unused-model',
                               '--work-dir','unused-work','--layers','0','--dry-run'],text=True,capture_output=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('partial-layer',result.stderr)

if __name__=='__main__':
    unittest.main()
