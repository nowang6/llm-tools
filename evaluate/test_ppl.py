import torch
import os
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer

path = os.getenv("MODEL_ID")
tokenizer = LlamaTokenizer.from_pretrained(path,trust_remote_code=True)
model = LlamaForCausalLM.from_pretrained(path,trust_remote_code=True)

inputs = tokenizer("The capital of the United States is Washington, D.C", return_tensors = "pt")
loss = model(input_ids = inputs["input_ids"], labels = inputs["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl)

inputs_wiki_text = tokenizer("The capital of the United States is New York", return_tensors = "pt")
loss = model(input_ids = inputs_wiki_text["input_ids"], labels = inputs_wiki_text["input_ids"]).loss
ppl = torch.exp(loss)
print(ppl)
