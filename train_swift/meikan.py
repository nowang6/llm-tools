import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import DatasetName, ModelType, SftArguments, sft_main, llm_sft

sft_args = SftArguments(
    model_type=ModelType.qwen1half_7b_chat,
    model_id_or_path='/home/niwang/models/Qwen15-14B-Chat',
    dataset="/home/niwang/code/meikang-llm-train/data/entity_sft_data.json",
    max_length=2048,
    learning_rate=1e-4,
    output_dir='output',
    lora_target_modules=['ALL'])
output = sft_main(sft_args)
best_model_checkpoint = output['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')