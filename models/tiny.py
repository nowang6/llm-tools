import torch
from transformers import pipeline
import os

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


pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])