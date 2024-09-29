from opencompass.models import OpenAI
from mmengine.config import read_base
from opencompass.cli.main import main

with read_base():
    from opencompass.configs.datasets.demo.demo_gsm8k_chat_gen import gsm8k_datasets
    from opencompass.configs.datasets.demo.demo_math_chat_gen import math_datasets

datasets = gsm8k_datasets + math_datasets
api_meta_template = dict(round=[
    dict(role='HUMAN', api_role='HUMAN'),
    dict(role='BOT', api_role='BOT', generate=True),
], )
models = [
    dict(
        abbr='glm4-9b-chat',
        type=OpenAI,
        path='glm4-9b-chat',
        openai_api_base = "http://xxxx:18080/v1/chat/completions",
        key='empty',  
        meta_template=api_meta_template,
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=4096,
        batch_size=8
    )
]


# opencompass evaluate/open_compass_eval.py--debug
