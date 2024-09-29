
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizerFast, LlamaForCausalLM
import torch
from torch.nn import CrossEntropyLoss


model_id = "/home/niwang/models/Meta-Llama-3-8B-Instruct"

tokenizer = LlamaTokenizerFast.from_pretrained(model_id)
model = LlamaForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})



sentences = ["who are you?", "who are you?"]


inputs = tokenizer(sentences, padding='max_length', max_length=50, truncation=True, return_tensors="pt").to(model.device)



inputs.keys()

# batch_size, sequence_length
bs, sl = inputs['input_ids'].size()


outputs = model.forward(**inputs, labels=inputs['input_ids'])
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
print(ppls)
