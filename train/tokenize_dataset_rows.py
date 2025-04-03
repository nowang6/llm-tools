import argparse
import json
from tqdm import tqdm
import datasets
import transformers

from datasets.arrow_dataset import Dataset

def preprocess(tokenizer, config, example, max_seq_length):
    #问题
    prompt = example["context"]
    #答案
    target = example["target"]
    #问题分词
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    #答案分词
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    #input_ids:问题分词+答案分词  seq_len:答案长度
    return {"input_ids": input_ids, "seq_len": len(prompt_ids)}


def read_jsonl(path, max_seq_length, skip_overlength=False):
    model_name = "/home/niwang/models/chatglm2-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, device_map='auto')
    with open(path, "r",encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            feature = preprocess(tokenizer, config, example, max_seq_length)
            if skip_overlength and len(feature["input_ids"]) > max_seq_length:
                continue
            feature["input_ids"] = feature["input_ids"][:max_seq_length]
            yield feature


def main():
    jsonl_path="data/wenlv_data.jsonl"
    save_path="data/wenlv_data"
    max_seq_length=384
    skip_overlength=True

    dataset = datasets.Dataset.from_generator(
        lambda: read_jsonl(jsonl_path, max_seq_length, skip_overlength)
    )
    dataset.save_to_disk(save_path)

if __name__ == "__main__":
    main()