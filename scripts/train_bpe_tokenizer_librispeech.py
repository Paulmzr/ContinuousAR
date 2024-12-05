from datasets import load_dataset, concatenate_datasets
from tokenizers import ByteLevelBPETokenizer
import os





train_dataset = load_dataset("WillHeld/librispeech_parquet", split="train.960")

# 提取文本列
def batch_iterator(batch_size=1000):
    for i in range(0, len(train_dataset), batch_size):
        yield train_dataset[i: i + batch_size]["text"]

# 初始化 ByteLevel BPE Tokenizer
tokenizer = ByteLevelBPETokenizer()

# 训练 Tokenizer
tokenizer.train_from_iterator(batch_iterator(), vocab_size=5000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
])
save_dir = "bpe_tokenizer_librispeech"
os.makedirs(save_dir, exist_ok=True)

# 保存 Tokenizer
tokenizer.save_model(save_dir)
