from datasets import load_dataset
from g2p_en import G2p
import string

# 加载 Hugging Face 数据集
dataset = load_dataset("WillHeld/librispeech_parquet")

# 初始化 G2p
g2p = G2p()
ignored = {" ", *string.punctuation}  # 定义需要忽略的符号

# 定义函数来生成音素
def add_phoneme_column(example):
    text = example['text']  # 假设数据集中有 'text' 列
    phones = g2p(text)  # 使用 g2p 转换文本为音素
    phoneme = " ".join(["_" if p in ignored else p for p in phones])  # 转换为音素并去除不需要的符号
    example['phoneme'] = phoneme  # 将 phoneme 添加到数据行
    return example

for split_name in dataset.keys():
    print(f"Processing split: {split_name}")
    dataset[split_name] = dataset[split_name].map(add_phoneme_column)

# 打印处理后的数据集前几行来检查效果
print(dataset['train.960'][0])  # 假设数据集有一个 "train" split
print(dataset['test'][0])

# 保存处理后的数据集到本地
dataset.save_to_disk('~/librispeech_phoneme')
