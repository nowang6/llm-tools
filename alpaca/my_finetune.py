import os
import sys
from typing import List

import torch
import transformers
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# 禁止huggingface联网，加快加载本地数据集的速度
os.environ['HF_DATASETS_OFFLINE'] = '1'

base_model = "/home/niwang/models/llama-7b-hf"
data_path = "/home/niwang/data/alpaca-cleaned/alpaca_data_cleaned.json"
output_dir = "/home/niwang/data/llama-7b-hf-trained"
# training hyperparams
batch_size = 128
micro_batch_size = 4 # 单张卡上的batch_size，一次梯度的batch_size
num_epochs = 3
learning_rate = 3e-4
cutoff_len = 256
val_set_size = 2000
# lora hyperparams
lora_r = 8  # rank
lora_alpha = 16 # lora对于预测的影响
lora_dropout = 0.05
lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
] # 对哪些层添加lora, 还有k, o
# llm hyperparams
train_on_inputs = False  # if False, masks out inputs in loss 在input上也要做训练
add_eos_token = True
group_by_length = False  # faster, but produces an odd training loss curve
# wandb params
wandb_project = ""
wandb_run_name = ""
wandb_watch = ""  # options: false | gradients | all
wandb_log_model = ""  # options: false | true
resume_from_checkpoint = None  # either training checkpoint or final adapter
prompt_template_name = "alpaca"  # The prompt template to use, will default to alpaca.


tokenizer = LlamaTokenizer.from_pretrained(base_model,trust_remote_code=True) #加载token
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (0)


def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len, # 上下文长度 1k, 8k
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

prompt_no_input = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
prompt_input = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

def generate_and_tokenize_prompt(data_point):
  instruction = data_point["instruction"]
  input = data_point["input"]
  output = data_point["output"]
  full_prompt = ""
  if input:
    full_prompt = prompt_input.format(instruction=instruction, input=input)
  else:
    full_prompt = prompt_no_input.format(instruction=instruction)
  user_prompt = full_prompt
  full_prompt = full_prompt + output
  
  tokenized_full_prompt = tokenizer(full_prompt,truncation=True,max_length=cutoff_len,padding=False,return_tensors=None)
  tokenized_full_prompt["input_ids"].append(tokenizer.eos_token_id)
  tokenized_full_prompt["attention_mask"].append(1)
  
  tokenized_user_prompt = tokenizer(user_prompt,truncation=True,max_length=cutoff_len,padding=False,return_tensors=None)
  user_prompt_len = len(tokenized_user_prompt["input_ids"])
  if add_eos_token:
      user_prompt_len -= 1

  tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["input_ids"][user_prompt_len:]
  return tokenized_full_prompt


data = load_dataset("json", data_files=data_path)
train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)

device_map = "auto"
model = LlamaForCausalLM.from_pretrained(base_model,load_in_8bit=False,torch_dtype=torch.float16,device_map=device_map) #加载模型

config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=lora_target_modules,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        warmup_steps=100,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if val_set_size > 0 else None,
        save_steps=200,
        output_dir=output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if val_set_size > 0 else False,
        group_by_length=group_by_length,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False

model = torch.compile(model)

trainer.train()

model.save_pretrained(output_dir)
