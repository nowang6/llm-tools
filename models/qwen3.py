from transformers import Qwen3ForCausalLM, AutoTokenizer, Qwen3MoeForCausalLM
import torch.nn.functional as F
import torch

model_path = "/data/models/Qwen2.5-7B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen3MoeForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
#prompt = "今天天气不错，我想去"
prompt = "苹果公司（Apple Inc.）的首席执行官（CEO）是蒂姆·库克，鼎桥通信技术有限公司的首席执行官（CEO）是"
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

# forward pass to get next token prediction
outputs = model.forward(
    input_ids=model_inputs.input_ids,          # token ids of input text
    attention_mask=model_inputs.attention_mask, # attention mask to handle padding
    use_cache=True,                            # enable key/value caching for faster generation
    output_attentions=False,                   # don't output attention weights
    output_hidden_states=True                  # output all layer's hidden states
)

# Get embeddings from the first hidden state
# hidden_states[0] is the embedding layer output
initial_embeddings = outputs.hidden_states[0]
print(f"Input embeddings shape (from hidden states): {initial_embeddings.shape}")

# Get the last layer's hidden states
last_layer_hidden = outputs.hidden_states[-1]  # shape: [batch_size, sequence_length, hidden_size]
print(f"Last layer hidden state shape: {last_layer_hidden.shape}")

# Get the hidden state for the last token
last_token_hidden = last_layer_hidden[:, -1, :]  # shape: [batch_size, hidden_size]
print(f"Last token hidden state shape: {last_token_hidden.shape}")

# Convert hidden states to logits using model's lm_head
logits_from_hidden = model.lm_head(last_token_hidden)  # shape: [batch_size, vocab_size]
print(f"\nLogits shape (from hidden states): {logits_from_hidden.shape}")

# apply softmax to convert logits to probabilities (using logits from hidden states)
probs = F.softmax(logits_from_hidden, dim=-1)

# get top 5 tokens and their probabilities
top_k = 20
top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)

# print top 5 predictions
print("\nTop 5 predictions:")
for i in range(top_k):
    token_id = top_indices[0, i]  # take first batch
    token_prob = top_probs[0, i]  # take first batch
    token_text = tokenizer.decode(token_id)
    print(f"Token: '{token_text}', Probability: {token_prob:.4f}")



