from collections import Counter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
import json
from tqdm import tqdm


dataset = load_from_disk('/data/mazhengrui/librispeech_phoneme')['train.960']


# 提取所有的 phoneme
def extract_phonemes_from_dataset(dataset):
    phoneme_counter = Counter()

    # 遍历数据集中的每一个样本
    for sample in tqdm(dataset):
        phonemes_list = sample["phoneme"].split()
        phoneme_counter.update(phonemes_list)
    
    # 获取所有唯一的音素
    unique_phonemes = list(phoneme_counter.keys())
    with open('/data/mazhengrui/SpeechLLaMA/phoneme_tokenizer_librispeech/phoneme_counter.json', 'w') as f:
        json.dump(dict(phoneme_counter), f)
    return unique_phonemes

phonemes = extract_phonemes_from_dataset(dataset)

# 从4开始编号，以预留特殊标记的ID
vocab = {phoneme: i + 4 for i, phoneme in enumerate(phonemes)}
vocab["<s>"] = 0    # 添加 [BOS] token 并设置为0
vocab["<pad>"] = 1  # 添加 [PAD] token 并设置为1
vocab["</s>"] = 2   # 添加 [EOS] token 并设置为2
vocab["<unk>"] = 3  # 添加 [UNK] token 并设置为3

# 创建 Tokenizer
tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

# **设置后处理器，自动添加 <s> 和 </s>**
tokenizer.post_processor = TemplateProcessing(
    single="<s> $0 </s>",
    pair=None,
    special_tokens=[
        ("<s>", vocab["<s>"]),
        ("</s>", vocab["</s>"]),
    ],
)

# 定义特殊标记
special_tokens = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>"
}

# 转换为 PreTrainedTokenizerFast
tokenizer_fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer,**special_tokens)
tokenizer_fast.padding_side = "left"



# 保存 tokenizer
tokenizer_fast.save_pretrained("/data/mazhengrui/SpeechLLaMA/phoneme_tokenizer_librispeech")



