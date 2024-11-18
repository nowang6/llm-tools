
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI
import json

# 初始化 llm
llm = OpenAI(
    api_key="empty",
    base_url="http://192.168.1.250:8000/v1"
)
model = "Qwen25-Coder"


def load_human_eval_dataset():
    dataset = load_dataset("/Users/niwang/datasets/openai_humaneval")
    return dataset['test']

def query_llm(prompt):
    completion = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "请直接输出Python代码"},
            {"role": "user", "content": prompt}
        ]
    )
    message = completion.choices[0].message
    return message.content


def test_llm_code(llm_gen_code, test_code,entry_point):
    try:
        namespace = {}
        exec(llm_gen_code, namespace)
        exec(test_code, namespace)
        exec(f"check({entry_point})", namespace)
        namespace.clear()
        return "Pass"
    except:
        return "Fail"

def clean_markdown(code):
    code = code.replace('```python', '')
    code = code.replace('```', '')
    return code


if __name__ == '__main__':
    dataset = load_human_eval_dataset()
    datas = []
    count = 0
    for data in dataset:
        count += 1
        prompt = data['prompt']
        entry_point = data['entry_point']
        llm_gen_code = query_llm(prompt)
        llm_gen_code = clean_markdown(llm_gen_code)
        test_code = data['test']
        res = test_llm_code(llm_gen_code, test_code, entry_point)
        data["result"] = res
        datas.append(data)
    with open('llm_eval_res.json', 'w') as f:
        json.dump(datas, f, indent=4)
        
