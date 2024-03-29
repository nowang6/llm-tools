from transformers import LlamaTokenizer, LlamaForCausalLM

path = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(path,trust_remote_code=True)
layer0 = model.model.layers[0]
print(layer0)
print("--------------")
for name, paramers in layer0.named_parameters():
    print(f"{name} have pramerter: {paramers.numel()}")

