"""Rebuild the recorded calibration selection from user-provided dataset files."""
import argparse
import hashlib
import json
from pathlib import Path

def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gsm8k-train',type=Path,required=True)
    p.add_argument('--mbpp-full',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError('Refusing to replace an existing calibration file')
    selection=json.loads((Path(__file__).resolve().parents[1]/'configs/calibration_selection.json').read_text())
    gsm=read(a.gsm8k_train)
    mbpp={str(row['task_id']):row for row in read(a.mbpp_full)}
    records=[]
    for item in selection:
        index=item['source_id'].rsplit('_',1)[1]
        if item['dataset']=='gsm8k':
            row=gsm[int(index)]
            question,answer=row['question'],row['answer']
            text='### Problem\n'+question+'\n\n### Solution\n'+answer
        else:
            row=mbpp[index]
            question,answer=row['text'],row['code']
            text='### Python Programming Task\n'+question+'\n\n### Reference Implementation\n```python\n'+answer+'\n```'
        if hashlib.sha256(text.encode()).hexdigest()!=item['text_sha256']:
            raise ValueError('Dataset content/version mismatch: '+item['source_id'])
        records.append(dict(text=text,prompt_text=question,source_id=item['source_id'],
                            dataset=item['dataset'],split='train',bucket=item['bucket']))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records),encoding='utf-8')
    print(json.dumps({'records':len(records),'content_hashes_verified':True}))

if __name__=='__main__':
    main()
