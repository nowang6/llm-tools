import promptbench as pb
from tqdm import tqdm
import os
import torch

# 设置代理地址和端口号
proxy_address = "127.0.0.1"
proxy_port = "7890"

# 设置HTTP代理
os.environ["HTTP_PROXY"] = f"http://{proxy_address}:{proxy_port}"
os.environ["HTTPS_PROXY"] = f"http://{proxy_address}:{proxy_port}"

# 可选：设置FTP代理
os.environ["FTP_PROXY"] = f"ftp://{proxy_address}:{proxy_port}"

# 可选：设置SOCKS代理
os.environ["SOCKS_PROXY"] = f"socks://{proxy_address}:{proxy_port}"

dataset = pb.DatasetLoader.load_dataset("sst2")
device = torch.device('cuda:1')

# load a model, flan-t5-large, for instance.
model = pb.LLMModel(model='llama2-7b', model_dir="/home/niwang/models/llama-7b-hf",device=device,max_new_tokens=10, temperature=0.0001)

prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
                     "Determine the emotion of the following sentence as positive or negative: {content}"
                     ])

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)



for prompt in prompts:
    preds = []
    labels = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['label']
        raw_pred = model(input_text)
        # process output
        pred = pb.OutputProcess.cls(raw_pred, proj_func)
        preds.append(pred)
        labels.append(label)
    
    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")
