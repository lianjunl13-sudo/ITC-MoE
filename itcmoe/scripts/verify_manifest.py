"""Verify source-package SHA256 checksums without loading a model."""
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    failed=[];count=0
    for line in (ROOT/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        expected,name=line.split('  ',1)
        path=(ROOT/name).resolve()
        path.relative_to(ROOT.resolve())
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            failed.append(name)
        count+=1
    if failed:raise RuntimeError('File verification failed: '+', '.join(failed))
    print(f'SHA256 verification passed for {count} files')

if __name__=='__main__':main()
