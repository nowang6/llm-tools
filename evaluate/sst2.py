from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.prompts import PromptTemplate


path = 'openbmb/MiniCPM-2B-sft-fp32'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, device_map='cuda', trust_remote_code=True)


prompt_template = PromptTemplate.from_template(
    "Classify the sentence as positive or negative, just output positive or negative: {sentence}"
)

datasets = load_dataset("glue", "sst2")
val_datasets = datasets["validation"]

total, correct = 0, 0 
for line in val_datasets:
  sentence = line['sentence']
  label = line['label']
  prompt =  prompt_template.format(sentence=sentence)
  pred = model.chat(tokenizer, prompt, temperature=0.8, top_p=0.8)
  mapping = {
        "positive": 1,
        "negative": 0
    }
  pred = mapping.get(pred, -1)
  if pred == label:
    correct += 1
  total += 1

print(f'{correct} of {total} is correct, accuracy rate is {correct/total}')
