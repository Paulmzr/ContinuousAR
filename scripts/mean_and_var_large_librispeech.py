import argparse
import logging

import torch

from transformers import AutoProcessor
from transformers import EncodecModel
import pdb


from datasets import load_dataset, concatenate_datasets, Audio
import random
from tqdm import tqdm

logger = logging.getLogger(__name__)

SAMPLING_RATE=24000




def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)

    

    # Initialize the model and tokenizer
    codec = EncodecModel.from_pretrained("facebook/encodec_24khz", torch_dtype=torch_dtype)
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    codec.to(device)
    
    
    train_dataset = load_dataset("WillHeld/librispeech_parquet", split="train.960")
    train_dataset = train_dataset.select_columns(['audio'])
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    # 随机选择1000个样本
    num_samples = 10000
    random_indices = random.sample(range(len(train_dataset)), num_samples)
    selected_data = train_dataset.select(random_indices)

    def filter_function(example):
        return (len(example["audio"]["array"]) / 320) < 2048
    
    
    #logger.info(f"original train dataset: {len(train_dataset)} samples.")
    #selected_data = train_dataset.filter(filter_function)
    #logger.info(f"filtered train dataset: {len(selected_data)} samples.")
    
    
    # 设置batch size为10
    batch_size = 20
    num_batches = len(selected_data) // batch_size
    
    all_embeds = []
    
    for i in tqdm(range(num_batches)):
        batch_data = selected_data.select(range(i * batch_size, (i + 1) * batch_size))
        batch_list = list(batch_data)
        
        audio_arrays = [instance["audio"]["array"] for instance in batch_list]
        audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
            
        encoder_outputs = codec.encode(audio_inputs["input_values"].to(torch_dtype).to(device), audio_inputs["padding_mask"].to(device), bandwidth=6) #1,b,r,t, 1 due to one chunk
        speech_inputs_embeds = codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
        speech_attention_mask = audio_inputs["padding_mask"][..., ::320]
        assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
        speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
        
        speech_inputs_embeds = speech_inputs_embeds.view(-1, speech_inputs_embeds.size(-1)) # b*t, d
        speech_attention_mask = speech_attention_mask.contiguous().view(-1) # b*t

        selected_embeds = speech_inputs_embeds[speech_attention_mask == 1]
        all_embeds.append(selected_embeds.cpu())
        del selected_embeds
        
    all_embeds_concat = torch.cat(all_embeds, dim=0)
    
        # 计算均值和标准差
    mean = all_embeds_concat.mean(dim=0)
    std = all_embeds_concat.std(dim=0)
    var = all_embeds_concat.var(dim=0)

    print("Mean:", mean)
    print("Std:", std)
    print("Var:", var)

    torch.save(mean, 'mean.pt')
    torch.save(std, 'std.pt')
    
    return


if __name__ == "__main__":
    main()