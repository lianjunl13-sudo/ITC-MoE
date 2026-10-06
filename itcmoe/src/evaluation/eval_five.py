import torch
from itcmoe_timed import ITCMoETimed
from opencompass.datasets import GSM8KDataset,MATHEvaluator,MBPPEvaluator,SanitizedMBPPDataset,gsm8k_dataset_postprocess,math_postprocess_v2
from arithmetic import NumericEvaluator,ScreenMathDataset,last_number
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.partitioners import NaivePartitioner,NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask,OpenICLInferTask

TASK=__import__('os').environ['ITCMOE_FULL_TASK']
N={'mbpp':257,'gsm8k':1319,'multiarith':600,'singleop':562,'singleq':109}[TASK]
DATA_ROOT=__import__('os').environ['ITCMOE_DATA_DIR']
if TASK=='mbpp':
    dtype=SanitizedMBPPDataset;path=f'{DATA_ROOT}/sanitized-mbpp.jsonl'
    reader_cfg=dict(input_columns=['text','test_list'],output_column='test_list_2')
    prompt='You are an expert Python programmer, and here is your task:\n{text}\nYour code should pass these tests:\n\n{test_list}\n You should submit your final solution in the following format: ```python\n\n```'
    eval_cfg=dict(evaluator=dict(type=MBPPEvaluator),pred_role='BOT')
elif TASK=='gsm8k':
    dtype=GSM8KDataset;path='opencompass/gsm8k'
    reader_cfg=dict(input_columns=['question'],output_column='answer')
    prompt='{question}\nPlease reason step by step, and put your final answer within \\boxed{}.'
    eval_cfg=dict(evaluator=dict(type=MATHEvaluator,version='v2'),pred_postprocessor=dict(type=math_postprocess_v2),dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))
else:
    dtype=ScreenMathDataset;path=f'{DATA_ROOT}/{TASK}_full{N}.jsonl'
    reader_cfg=dict(input_columns=['question'],output_column='answer');prompt='{question}'
    eval_cfg=dict(evaluator=dict(type=NumericEvaluator),pred_postprocessor=dict(type=last_number))
datasets=[dict(abbr=f'{TASK}_full{N}',type=dtype,path=path,reader_cfg=reader_cfg,
    infer_cfg=dict(prompt_template=dict(type=PromptTemplate,template=dict(round=[dict(role='HUMAN',prompt=prompt)])),
        retriever=dict(type=ZeroRetriever),inferencer=dict(type=GenInferencer,batch_size=1)),eval_cfg=eval_cfg)]
models=[dict(type=ITCMoETimed,abbr=__import__('os').environ['ITCMOE_FULL_ABBR'],path=__import__('os').environ['ITCMOE_FULL_MODEL'],
    run_cfg=dict(num_gpus=1),generation_kwargs=dict(mask_id=151669,gen_length=4096,block_length=32,denoising_steps=32,
        temperature=1.,top_k=1,top_p=1.,cfg_scale=0.,remasking='low_confidence',threshold=.95,speculative=True),
    model_kwargs=dict(torch_dtype=torch.float16,trust_remote_code=True))]
infer=dict(partitioner=dict(type=NumWorkerPartitioner,num_worker=1),runner=dict(type=LocalRunner,max_num_workers=1,
    keep_tmp_file=True,task=dict(type=OpenICLInferTask),retry=0))
eval=dict(partitioner=dict(type=NaivePartitioner,n=1),runner=dict(type=LocalRunner,task=dict(type=OpenICLEvalTask,dump_details=True)))
work_dir=__import__('os').environ['ITCMOE_FULL_WORK']
