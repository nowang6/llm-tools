
from docx import Document
from typing import List
import jsonlines
import os
import json
import markdown
import re
import pdfplumber
from pathlib import Path
from openai import OpenAI
import os
import torch
import pandas as pd
import hashlib
import torch
from transformers import BertTokenizer, GPT2LMHeadModel
from torch.nn import CrossEntropyLoss
import os


# 解析文件类
def parse_docx(file_path): 
    pages = []  
    try: 
        doc = Document(file_path)  
        for para in doc.paragraphs:
            pages.append(para.text)
    except Exception as e:
        print(f"An error occurred while parsing the file '{file_path}': {e}")
    return "\n".join(pages)


def parse_txt(file_path):
    contents = []
    with open(file_path,"r") as reader:
        for line in reader:
            contents.append(line.strip())
    return contents

def join_line_to_paragraph(lines):
    contents = []
    one_paragraph = ""
    for line in lines:
        if line.startswith(" "): # 新段落开始
            contents.append(one_paragraph)
            one_paragraph = line
        elif len(line) <=15: # 段落结束
            one_paragraph = one_paragraph + line
            contents.append(one_paragraph)
            one_paragraph=""
        else:
            one_paragraph = one_paragraph + line
    contents.extend(one_paragraph)
    return contents
    

def parse_md(file_path):
    # 读取Markdown文档内容
    with open(file_path, 'r') as reader:
        markdown_text = reader.read()
    # 将Markdown文档转换为HTML
    html = markdown.markdown(markdown_text)
    # 从HTML中提取纯文本
    text = re.sub('<[^<]+?>', '', html)  # 使用正则表达式删除HTML标签
    return join_line_to_paragraph(text.split("\n"))
        
    
def parse_pdf(file_path):
    pages = []
    try:
        with pdfplumber.open(file_path) as pdf_reader:
        # 遍历PDF文件的每一页
            content = ""
            for i, page in enumerate(pdf_reader.pages):
                # 提取页面文本
                text = page.extract_text()
                pages.append(text)
    except Exception as e:  
        print(f"无法打开文件 {file_path}: {e}")
    #处理段落，按照段落切分
    return "\n".join(pages)
   

def get_files(folder:str):
    files = []
    for root, dirs, file_names in os.walk(folder):
        for file_name in file_names:
            full_path = os.path.join(root, file_name)
            if full_path.endswith(".DS_Store"):
                continue
            files.append(full_path)
    return files


def get_video_files(folder:str):
    files = []
    for root, dirs, file_names in os.walk(folder):
        for file_name in file_names:
            full_path = os.path.join(root, file_name)
            if full_path.endswith(".mp4") or full_path.endswith(".avi") or full_path.endswith(".mkv") or full_path.endswith(".mov") or full_path.endswith(".wmv") :
                files.append(full_path)
    return files




def get_image_files(folder:str):
    image_extensions = [
    ".jpeg", ".jpg",  # JPEG/JPG
    ".png",           # PNG
    ".gif",           # GIF
    ".bmp",           # BMP
    ".tiff", ".tif",  # TIFF/TIF
    ".webp",          # WEBP
    ".svg",           # SVG
    ".heif", ".heic", # HEIF/HEIC
    ".cr2", ".nef", ".arw",  # RAW (example extensions)
    ".psd"            # PSD
    ]
    files = []
    for root, dirs, file_names in os.walk(folder):
        for file_name in file_names:
            full_path = os.path.join(root, file_name)
            file_name, file_extension = os.path.splitext(full_path)
            file_extension = file_extension.lower()
            if file_extension in image_extensions:
                files.append(full_path)
    return files

def get_file_size_in_kb(file_path):
    # 获取文件大小（字节）
    file_size_bytes = os.path.getsize(file_path)
    # 将文件大小转换为 MB
    file_size_mb = file_size_bytes / 1024 
    return file_size_mb

# json 读写
def texts_2_pretrain_json(texts:List, file_path):
    if os.path.exists(file_path):
        print(f"Already exit:{file_path}")
        return
    
    json_list = []
    for text in texts:
        json_list.append({"text":text})
    
    with open(file_path, 'w') as writer:
        json.dump(json_list, writer, ensure_ascii=False)


def dump_list_json_file(file, datas):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)


def load_list_json_file(file):
    datas = []
    with open(file,"r",encoding="utf-8") as file:
        datas = json.load(file)
    return datas     
        




def rename_file(original_file,new_file):
    os.rename(original_file,new_file)

def write_file(file,str):
    with open(file,"w",encoding="utf-8") as file:
        file.write(str)

def read_file(file):
    content = ""
    if not os.path.exists(file):
        return content
    with open(file,"r",encoding="utf-8") as reader:
        content = reader.read()
    return content

def write_to_csv(file_name, **kwargs):
    df = pd.DataFrame()
    for series_name, series_values in kwargs.items():
        df[series_name] = pd.Series(series_values)
    df.to_csv(file_name,index=False)
    

# 机器学习
def get_median_and_mean(nums):
    # 计算平均数
    mean = sum(nums) / len(nums)

    # 计算中位数
    sorted_numbers = sorted(nums)
    n = len(sorted_numbers)
    if n % 2 == 1:
        median = sorted_numbers[n // 2]
    else:
        median = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2

    return median, mean

def get_ppls(words):
    sentences = [f"{word}是一个有意义的人类可以理解的词语" for word in words]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "uer/gpt2-chinese-cluecorpussmall"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path,local_files_only=True).to(device)

    inputs = tokenizer(sentences, padding='max_length', max_length=50, truncation=True, return_tensors="pt").to(device)

    inputs.keys()

    # batch_size, sequence_length
    bs, sl = inputs['input_ids'].size()


    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs["logits"]
    logits.shape
    # 错位构造logits和label
    # 后续可用于计算交叉熵损失
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    

    # reshape成(bs*sl, vocab_size)
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    loss.shape

    # 计算平均损失，求平均时不计入padding
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    meanloss.shape

    # 计算ppl
    ppls = torch.exp(meanloss).cpu().numpy().tolist()
    return ppls

def calculate_md5(file_path, chunk_size=8192):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()