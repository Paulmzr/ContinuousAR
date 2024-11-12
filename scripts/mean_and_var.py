import argparse


import torch

from transformers import AutoProcessor
from transformers import EncodecModel
import pdb


from datasets import load_dataset, concatenate_datasets, Audio
import random



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
    
    
    train_dataset = load_dataset("mythicinfinity/libritts", "all", split=["train.clean.100", "train.clean.360", "train.other.500"])
    train_dataset = concatenate_datasets(train_dataset)
    train_dataset = train_dataset.select_columns(['audio'])
    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    selected_data = train_dataset.select(range(20))
    selected_list = list(selected_data)

    
    audio_arrays = [instance["audio"]["array"] for instance in selected_list]
    audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
        
    encoder_outputs = codec.encode(audio_inputs["input_values"].to(torch_dtype).to(device), audio_inputs["padding_mask"].to(device), bandwidth=6) #1,b,r,t, 1 due to one chunk
    speech_inputs_embeds = codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
    speech_attention_mask = audio_inputs["padding_mask"][..., ::320]
    assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
    speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
    
    speech_inputs_embeds = speech_inputs_embeds.view(-1, speech_inputs_embeds.size(-1)) # b*t, d
    speech_attention_mask = speech_attention_mask.contiguous().view(-1) # b*t

    selected_embeds = speech_inputs_embeds[speech_attention_mask == 1]
    
    mean = selected_embeds.mean(dim=0)
    std = selected_embeds.std(dim=0)
    var = selected_embeds.var(dim=0)


    torch.save(mean, 'mean.pt')
    torch.save(std, 'std.pt')
    
    print(mean)
    print(std)
    print(var)
    
    return


if __name__ == "__main__":
    main()