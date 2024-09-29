import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, inference_stream_vllm
)
import torch

model_type = ModelType.qwen1half_0_5b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen
# Experimental environment: 3090


model_type = ModelType.qwen1half_7b_chat_awq
llm_engine = get_vllm_engine(model_type, torch.float16, max_model_len=4096)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 512

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

# 流式
history1 = resp_list[1]['history']
query = '这有什么好吃的'
request_list = [{'query': query, 'history': history1}]
gen = inference_stream_vllm(llm_engine, template, request_list)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f"history: {resp_list[0]['history']}")

"""
query: 你好!
response: 你好！有什么问题我可以帮助你吗？
query: 浙江的省会在哪？
response: 浙江省的省会是杭州市。
query: 这有什么好吃的
response: 浙江有很多美食，以下列举一些具有代表性的：

1. 杭州菜：杭州作为浙江的省会，以其精致细腻、注重原汁原味而闻名，如西湖醋鱼、龙井虾仁、叫化童鸡等都是特色菜品。

2. 宁波汤圆：宁波的汤圆皮薄馅大，甜而不腻，尤其是冬至和元宵节时，当地人会吃宁波汤圆庆祝。

3. 温州鱼丸：温州鱼丸选用新鲜鱼类制作，口感弹滑，味道鲜美，常常配以海鲜煮食。

4. 嘉兴粽子：嘉兴粽子以其独特的三角形和咸甜两种口味著名，特别是五芳斋的粽子非常有名。

5. 金华火腿：金华火腿是中国著名的腌制肉类，肉质紧实，香味浓郁，常作为节日礼品。

6. 衢州烂柯山豆腐干：衢州豆腐干质地细腻，味道鲜美，是浙江的传统小吃。

7. 舟山海鲜：浙江沿海地带的舟山有丰富的海鲜资源，如梭子蟹、带鱼、乌贼等，新鲜美味。

以上只是部分浙江美食，浙江各地还有许多特色小吃，你可以根据自己的口味去尝试。
history: [('浙江的省会在哪？', '浙江省的省会是杭州市。'), ('这有什么好吃的', '浙江有很多美食，以下列举一些具有代表性的：\n\n1. 杭州菜：杭州作为浙江的省会，以其精致细腻、注重原汁原味而闻名，如西湖醋鱼、龙井虾仁、叫化童鸡等都是特色菜品。\n\n2. 宁波汤圆：宁波的汤圆皮薄馅大，甜而不腻，尤其是冬至和元宵节时，当地人会吃宁波汤圆庆祝。\n\n3. 温州鱼丸：温州鱼丸选用新鲜鱼类制作，口感弹滑，味道鲜美，常常配以海鲜煮食。\n\n4. 嘉兴粽子：嘉兴粽子以其独特的三角形和咸甜两种口味著名，特别是五芳斋的粽子非常有名。\n\n5. 金华火腿：金华火腿是中国著名的腌制肉类，肉质紧实，香味浓郁，常作为节日礼品。\n\n6. 衢州烂柯山豆腐干：衢州豆腐干质地细腻，味道鲜美，是浙江的传统小吃。\n\n7. 舟山海鲜：浙江沿海地带的舟山有丰富的海鲜资源，如梭子蟹、带鱼、乌贼等，新鲜美味。\n\n以上只是部分浙江美食，浙江各地还有许多特色小吃，你可以根据自己的口味去尝试。')]
"""