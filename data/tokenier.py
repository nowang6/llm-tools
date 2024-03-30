from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path


data_dir = "data/xiyou"
txt_path = [x for x in Path(data_dir).glob("*.txt")]

SentencePieceTrainer.Train(input=txt_path,
                           vocab_size = 4096,
                           model_prefix = "models/m",
                           model_type = "unigram"
                           )

m = SentencePieceProcessor(model_file = "models/m.model")

print(m)
