import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def command(self, work):
        return [sys.executable, str(ROOT/'scripts/run_pipeline.py'), '--original-model', str(work/'original'),
                '--work-dir', str(work/'output'), '--calibration-jsonl', str(work/'calibration.jsonl'),
                '--mbpp-source-jsonl', str(work/'mbpp.jsonl'), '--opencompass-root', str(work/'opencompass'),
                '--data-dir', str(work/'data'), '--dry-run']

    def test_default_pipeline_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            result = subprocess.run(self.command(work), capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{"stage":')]
            self.assertEqual(len(rows), 11)
            self.assertEqual([r['dataset'] for r in rows[-5:]], ['mbpp', 'gsm8k', 'multiarith', 'singleop', 'singleq'])
            self.assertFalse((work/'output').exists())

    def test_reversed_stages_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            config = work/'invalid.json'
            config.write_text(json.dumps({'method': 'itcmoe', 'budget': 10,
                                         'stages': ['hot', 'rank'], 'datasets': ['mbpp']}), encoding='utf-8')
            result = subprocess.run(self.command(work)+['--config', str(config)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((work/'output').exists())


if __name__ == '__main__':
    unittest.main()
