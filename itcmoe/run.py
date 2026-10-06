"""itcmoe stage entry point; model and data paths are supplied on the command line."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'covariance', 'rank-inputs', 'rank', 'prepare-hot', 'hot', 'fit-hot', 'export-hot', 'evaluate'])
    p.add_argument('--budget', type=int, choices=[10, 20, 30], default=10)
    p.add_argument('--original-model', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--data-dir', type=Path)
    p.add_argument('--calibration-jsonl', type=Path)
    p.add_argument('--mbpp-source-jsonl', type=Path)
    p.add_argument('--dataset', choices=['mbpp','gsm8k','multiarith','singleop','singleq'], default='mbpp')
    p.add_argument('--opencompass-root', type=Path)
    p.add_argument('--layers', type=int, nargs='+')
    p.add_argument('--factor-library', type=Path)
    p.add_argument('--reuse-calibration-from', type=Path)
    p.add_argument('--operator', choices=['on','off'], default='on')
    p.add_argument('--hot-compensation', choices=['on','off'], default='on')
    p.add_argument('--candidate', choices=['on','off'], default='on')
    p.add_argument('--candidate-size', type=int, default=48)
    p.add_argument('--shrinkage-eta', type=float, default=.25)
    p.add_argument('--anchor-lambda', type=float, default=.1)
    p.add_argument('--hot-distance', type=int, default=3)
    p.add_argument('--hot-confidence', type=float, default=.7)
    p.add_argument('--run-name', default='default')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def normalize(text):
    return ' '.join(text.strip().split())

def file_sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()

def prepare(a):
    if a.reuse_calibration_from:
        source=a.reuse_calibration_from.resolve()
        if source==a.work_dir:
            raise ValueError('The calibration source must differ from the current work directory')
        source_identity=source/'run_identity.json'
        if source_identity.exists() and json.loads(source_identity.read_text())['original_model']!=str(a.original_model):
            raise ValueError('The calibration source refers to another original model path; verify teacher identity')
        for name in ('reference/calibration.jsonl','reference/covariance_reference.pt','base_training/rank_routed.pt'):
            if not (source/name).is_file():raise ValueError('Missing file in calibration source: '+name)
        for name in ('reference','base_training'):
            (a.work_dir/name).mkdir(parents=True,exist_ok=True)
        selected=list((source/'reference').iterdir())+[source/'base_training/rank_routed.pt']
        for path in selected:
            if not path.is_file():continue
            target=a.work_dir/('base_training' if path.name=='rank_routed.pt' else 'reference')/path.name
            if target.exists():
                if file_sha(target)!=file_sha(path):
                    raise ValueError('Existing calibration file differs from the source: '+path.name)
                continue
            if path.suffix=='.pt':
                try:os.link(path,target)
                except OSError:shutil.copy2(path,target)
            else:shutil.copy2(path,target)
        dump(a.work_dir/'calibration_reuse.json',dict(source=str(source),same_covariance=True,same_rank_inputs=True))
        return
    if not a.calibration_jsonl:
        raise ValueError('--calibration-jsonl is required for prepare')
    expected = json.loads((PACKAGE/'configs/calibration_selection.json').read_text(encoding='utf-8'))
    raw = [json.loads(line) for line in a.calibration_jsonl.read_text(encoding='utf-8').splitlines() if line.strip()]
    rows = {row['source_id']: row for row in raw}
    clean = []
    for item in expected:
        row = rows[item['source_id']]
        if any(row[k]!=item[k] for k in ('dataset','split','bucket')):
            raise ValueError('Calibration split metadata mismatch: '+item['source_id'])
        actual = hashlib.sha256(row['text'].encode()).hexdigest()
        if actual != item['text_sha256']:
            raise ValueError('Calibration content does not match the recorded source selection: '+item['source_id'])
        clean.append({k: row[k] for k in ('text','prompt_text','source_id','dataset','split','bucket')})
    reference = a.work_dir/'reference'
    reference.mkdir(parents=True, exist_ok=True)
    (reference/'calibration.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in clean), encoding='utf-8')
    cfg = json.loads((a.original_model/'config.json').read_text())
    shape = (cfg['num_hidden_layers'],cfg['num_experts'],cfg['hidden_size'],cfg['moe_intermediate_size'])
    if shape != (48,128,2048,768):
        raise ValueError('This release profile supports the recorded SDAR shape only')
    if cfg.get('td_moe_enabled', False):
        raise ValueError('The source must be the original uncompressed model')
    for li in range(48):
        if not (a.original_model/f'layer-{li}-ep-0-of-1.safetensors').is_file():
            raise ValueError('Expected layer-sharded original checkpoint; arbitrary HF reshaping is not silently performed')
    if not (a.original_model/'others.safetensors').is_file():
        raise ValueError('Missing non-expert original shard: others.safetensors')
    dump(reference/'input_manifest.json', {'budget':a.budget,'shape':shape,'calibration_records':len(clean)})

def prepare_hot(a):
    base = a.work_dir/'base_model'
    target = a.work_dir/'model'
    hot = a.work_dir/'hot'
    hot.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((base/'config.json').read_text())
    if cfg.get('hot_svd_rank',0):
        raise ValueError('Expected an uncompensated base model')
    if not a.mbpp_source_jsonl:
        raise ValueError('--mbpp-source-jsonl is required to reconstruct calibration prompts')
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('The compensation output directory must be empty')
    target.mkdir(parents=True, exist_ok=True)
    for path in base.iterdir():
        if not path.is_file():
            continue
        dest = target/path.name
        if dest.exists():
            raise FileExistsError('Refusing to overwrite an existing compensation view: '+str(dest))
        if path.suffix == '.safetensors':
            os.link(path, dest)
        else:
            shutil.copy2(path, dest)
    cfg['hot_svd_rank'] = 0
    dump(target/'config.json',cfg)
    mbpp = {str(r['task_id']):r for r in map(json.loads,a.mbpp_source_jsonl.read_text(encoding='utf-8').splitlines())}
    source = [json.loads(line) for line in (a.work_dir/'reference/calibration.jsonl').read_text(encoding='utf-8').splitlines()]
    rng = random.Random(16037)
    selected = []
    for bucket in ('gsm8k_train','mbpp_train'):
        rows = [r for r in source if r['bucket']==bucket and r['split']=='train']
        rng.shuffle(rows)
        for i,r in enumerate(rows[:16]):
            if bucket == 'gsm8k_train':
                prompt = r['prompt_text']+'\nPlease reason step by step, and put your final answer within \\boxed{}.'
            else:
                m = mbpp[r['source_id'].rsplit('_',1)[1]]
                if m['text'].strip()!=r['prompt_text'].strip():
                    raise ValueError('MBPP source mismatch')
                prompt = 'You are an expert Python programmer, and here is your task:\n'+r['prompt_text']+'\nYour code should pass these tests:\n\n'+'\n'.join(m['test_list'])+'\n You should submit your final solution in the following format: ```python\n\n```'
            selected.append(dict(source_id=r['source_id'],bucket=bucket,split='train' if i<12 else 'dev',prompt=prompt))
    dump(hot/'prompts.json',selected[::2]+selected[1::2])

def resource_preflight(a):
    if a.stage in ('prepare','prepare-hot'):
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; this entry point will not start on a CPU-only instance')
    free,total=torch.cuda.mem_get_info()
    if free < 70*1024**3:
        raise RuntimeError('Conservative preflight requires at least 70 GiB free GPU memory for the full SDAR profile')
    limit=Path('/sys/fs/cgroup/memory.max')
    current=Path('/sys/fs/cgroup/memory.current')
    if limit.exists() and current.exists() and limit.read_text().strip()!='max':
        available=int(limit.read_text())-int(current.read_text())
        if available < 96*1024**3:
            raise RuntimeError('Conservative preflight requires at least 96 GiB cgroup memory headroom')
    required=160 if a.stage=='rank' else 8
    if shutil.disk_usage(a.work_dir).free < required*1024**3:
        raise RuntimeError(f'Insufficient disk headroom; this stage requires at least {required} GiB')

def commands(a):
    src = PACKAGE/'src'
    py = sys.executable
    layers = ['--layers', *map(str,a.layers)] if a.layers else []
    if a.stage=='covariance':
        return [[py,str(src/'tools/collect_covariance.py'),'--model-path',str(a.original_model),'--calib-jsonl',str(a.work_dir/'reference/calibration.jsonl'),'--output-cov-cache',str(a.work_dir/'reference/covariance_reference.pt'),'--resume','--shrinkage-eta',str(a.shrinkage_eta)]]
    if a.stage in ('rank-inputs','rank'):
        return [[py,str(src/f'training/rank_c{a.budget}.py'),'collect' if a.stage=='rank-inputs' else 'train',*layers]]
    if a.stage=='hot':
        if a.layers:
            raise ValueError('Full-model Hot calibration cannot run on a partial-layer export')
        return [[py,str(src/'compensation'/name),*args] for name,args in [
            ('collect_hot.py',['--round','1']),('build_svd.py',[]),('export_model.py',['--stage','initial']),
            ('collect_hot.py',['--round','2']),('fit_gains.py',['--anchor-lambda',str(a.anchor_lambda)]),('export_model.py',['--stage','final'])]]
    if a.stage=='fit-hot':
        return [[py,str(src/'compensation/fit_gains.py'),'--anchor-lambda',str(a.anchor_lambda),*layers]]
    if a.stage=='export-hot':
        return [[py,str(src/'compensation/export_model.py'),'--stage','final']]
    if a.stage=='evaluate':
        if not a.opencompass_root or not a.data_dir:
            raise ValueError('--opencompass-root and --data-dir are required')
        return [[py,str(a.opencompass_root/'run.py'),str(src/'evaluation/eval_five.py')]]
    return []

def main():
    a = arguments()
    if not 0 <= a.shrinkage_eta <= 1:
        raise ValueError("shrinkage eta must be finite and in [0, 1]")
    if not 8 <= a.candidate_size <= 128 or a.hot_distance < 0 or not 0 <= a.hot_confidence <= 1:
        raise ValueError('Invalid candidate size or Hot classification parameters')
    if not 0 <= a.anchor_lambda < float('inf'):
        raise ValueError('The anchor coefficient must be finite and nonnegative')
    if Path(a.run_name).name != a.run_name or a.run_name in ('.','..'):
        raise ValueError('The run name must be a single directory name')
    if a.layers and any(not 0 <= i < 48 for i in a.layers):
        raise ValueError('Layer indices must be between 0 and 47')
    a.original_model=a.original_model.resolve()
    a.work_dir=a.work_dir.resolve()
    env=dict(os.environ,ITCMOE_WORKDIR=str(a.work_dir),ITCMOE_REFERENCE=str(a.work_dir/'reference'),
             ITCMOE_ORIGINAL_MODEL=str(a.original_model),OMP_NUM_THREADS='8',PYTHONUNBUFFERED='1')
    env.update(ITCMOE_OPERATOR=a.operator, ITCMOE_HOT=a.hot_compensation, ITCMOE_CANDIDATE=a.candidate,
               ITCMOE_SHRINKAGE_ETA=str(a.shrinkage_eta), ITCMOE_CANDIDATE_SIZE=str(a.candidate_size), ITCMOE_HOT_DISTANCE=str(a.hot_distance),
               ITCMOE_HOT_CONFIDENCE=str(a.hot_confidence))
    if a.factor_library:
        env['ITCMOE_FACTOR_LIBRARY']=str(a.factor_library.resolve())
    if a.stage != 'evaluate' and (a.operator!='on' or a.hot_compensation!='on' or a.candidate!='on'):
        raise ValueError('Inference ablation switches apply only to evaluate; calibration must use the default method')
    if a.stage=='evaluate':
        env.update(ITCMOE_DATA_DIR=str(a.data_dir.resolve()),ITCMOE_FULL_TASK=a.dataset,
                   ITCMOE_EVAL_SEED='16037',
                   ITCMOE_FULL_MODEL=str(a.work_dir/'model'),ITCMOE_FULL_ABBR=f'itcmoe-c{a.budget}',
                   ITCMOE_FULL_WORK=str(a.work_dir/'evaluation'/a.run_name/a.dataset),
                   ITCMOE_SAMPLE_TIMING_PATH=str(a.work_dir/'evaluation'/a.run_name/a.dataset/'timing.jsonl'))
    paths=[str(PACKAGE/'src/evaluation'),str(PACKAGE/'src/training')]
    if a.opencompass_root:
        paths.append(str(a.opencompass_root.resolve()))
    env['PYTHONPATH']=os.pathsep.join(paths+[env.get('PYTHONPATH','')])
    plan=commands(a)
    if a.dry_run:
        print(json.dumps({'stage':a.stage,'budget':a.budget,'commands':plan,
            'runtime':dict(operator=a.operator,hot=a.hot_compensation,candidate=a.candidate,
                           candidate_size=a.candidate_size,hot_distance=a.hot_distance,
                           hot_confidence=a.hot_confidence),'anchor_lambda':a.anchor_lambda},indent=2))
        return
    a.work_dir.mkdir(parents=True,exist_ok=True)
    identity_file=a.work_dir/'run_identity.json'
    identity=dict(budget=a.budget,original_model=str(a.original_model),shrinkage_eta=a.shrinkage_eta)
    if identity_file.exists() and json.loads(identity_file.read_text())!=identity:
        raise RuntimeError('This work directory belongs to another budget or original model; use a separate directory')
    dump(identity_file,identity)
    if a.stage in ('rank', 'prepare'):
        cache = (a.reuse_calibration_from if a.stage == 'prepare' and a.reuse_calibration_from else a.work_dir)/'reference/covariance_reference.pt'
        if a.stage == 'rank' or (a.stage == 'prepare' and a.reuse_calibration_from):
            import torch
            sys.path.insert(0, str(PACKAGE/'src/tools'))
            from covariance_shrinkage import validate_shrinkage_cache
            validate_shrinkage_cache(torch.load(cache, map_location='cpu', weights_only=False), a.shrinkage_eta)
    resource_preflight(a)
    if a.stage=='prepare':
        prepare(a)
    elif a.stage=='prepare-hot':
        prepare_hot(a)
    else:
        for command in plan:
            subprocess.run(command,env=env,check=True)

if __name__=='__main__':
    main()
