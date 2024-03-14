import os  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
from langchain.prompts import PromptTemplate

# 禁止huggingface联网，加快加载本地数据集的速度
# os.environ['HF_DATASETS_OFFLINE'] = '1'

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


path = '/home/niwang/models/llama-7b-hf'
device = torch.device("cuda:0")


datasets = load_dataset("glue", "sst2")
# datasets = load_dataset("/home/niwang/data/glue-sst2")
val_datasets = datasets["validation"]


model = LlamaForCausalLM.from_pretrained(path, load_in_8bit=False,trust_remote_code=True).to(device) #加载模型
tokenizer = LlamaTokenizer.from_pretrained(path, trust_remote_code=True)

prompt_template = PromptTemplate.from_template(
    # "Classify the sentence as positive or negative, just output positive or negative: {sentence}"
    "Classify the sentence as positive or negative: {sentence}"
)

total, correct = 0, 0 
for line in val_datasets:
  sentence = line['sentence']
  label = line['label']
  prompt =  prompt_template.format(sentence=sentence)
  # prompt = "Hey, are you conscious? Can you talk to me?"
  
  inputs = tokenizer(prompt, return_tensors="pt")
  generate_ids = model.generate(inputs.input_ids.to(device), max_length=500)
  pred=tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  print(pred)
  mapping = {
        "positive": 1,
        "negative": 0
    }
  pred = mapping.get(pred, -1)
  if pred == label:
    correct += 1
  total += 1

print(f'{correct} of {total} is correct, accuracy rate is {correct/total}')
