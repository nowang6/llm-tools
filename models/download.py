import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2", torch_dtype=torch.bfloat16, device_map="auto")
