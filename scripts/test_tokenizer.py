from transformers import RobertaTokenizer

# 加载 BPE Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("/data/mazhengrui/SpeechLLaMA/bpe_tokenizer_libritts")

# 测试 Tokenizer
print(tokenizer.encode("Nice to meet you."))
