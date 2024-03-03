from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
import time

model_id = "/home/niwang/models/chatglm2-6b"
data_id = "/data/datas/BelleGroup/train_0.5M_CN"


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).half().cuda()
model = model.eval()

dataset = load_dataset(data_id)

sentences = dataset["train"]["instruction"]

# 测试代码输出 token 的速度
start_time_overall= time.time()
token_len_overall =0
for sentences in sentences[:100]:
    start_time = time.time()
    response, _ = model.chat(tokenizer, sentences, history=[])
    print(response)
    length = len(response)
    token_len_overall += length
    speed_one_sen = length / (time.time() - start_time)
    print(f'Speed of one sentence: {speed_one_sen}')

    speed_overall = token_len_overall / (time.time() - start_time_overall)
    print(f'Overall speed: {speed_overall}')
