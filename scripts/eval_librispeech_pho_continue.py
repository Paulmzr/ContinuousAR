import argparse
import logging
from typing import Tuple
from pathlib import Path

import torch
import torchaudio



from datasets import load_dataset, load_from_disk, Audio
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoProcessor
from accelerate.utils import set_seed
from speechllama.speech_llama_score import SpeechLlamaForCausalLM
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


import pdb


SAMPLING_RATE=24000

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def adjust_length_to_model(length, max_sequence_length):
    assert max_sequence_length > 0
    if length <= 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    return length


def filter_function(example):
    return ((len(example["audio"]["array"]) / 24000) < 10.0) and ((len(example["audio"]["array"]) / 24000) > 4.0)





def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=0)

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument(
        "--fp16",
        action="store_true",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    logger.warning(f"device: {device}, 16-bits inference: {args.fp16 or args.bf16}")

    if args.seed is not None:
        set_seed(args.seed)

    # Initialize the model and tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name_or_path)

    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
    model.infer_cfg = args.cfg
    model.initialize_codec("facebook/encodec_24khz") 
    model.initialize_scale("/data/mazhengrui/SpeechLLaMA/scale_encodec/librispeech/scale_10k")
    model.to(device)
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    eval_dataset = load_from_disk("/data/mazhengrui/librispeech_phoneme")["test"]
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    
    logger.info(f"original eval dataset: {len(eval_dataset)} samples.")
    eval_dataset = eval_dataset.filter(filter_function)
    logger.info(f"filtered eval dataset: {len(eval_dataset)} samples.")
    
    tokenized_eval_dataset = eval_dataset.map(
        lambda example: tokenizer(example["phoneme"]),
        batched=True,
    )
    

    batch_size = args.batch_size
    
    pathname = args.output + f".{args.ckpt}.cfg_{args.cfg}.seed_{args.seed}"
    output_path = Path("eval_continue") / pathname
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad(): 
        for i in range(0, len(tokenized_eval_dataset), batch_size):
            batch = tokenized_eval_dataset.select(range(i, min(i + batch_size, len(tokenized_eval_dataset))))
                    
            input_ids = [{"input_ids":instance["input_ids"]} for instance in batch]
            
            encodes = pad_without_fast_tokenizer_warning(
                tokenizer,
                input_ids,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt"
            )
            
            input_ids = encodes["input_ids"].to(device)
            attention_mask = encodes["attention_mask"].to(device)       
            text_input_length = input_ids.shape[1]

            audio_arrays = [instance["audio"]["array"] for instance in batch]
            audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
            
            
            encoder_outputs = model.codec.encode(audio_inputs["input_values"].to(model.dtype).to(device), audio_inputs["padding_mask"].to(device), bandwidth=6) #1,b,r,t, 1 due to one chunk
            speech_inputs_embeds = model.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
            #speech_inputs_embeds = self.codec.encode(audio_inputs["input_values"].to(self.model.dtype)).sample()  #b,d,t
            
            speech_attention_mask = audio_inputs["padding_mask"][..., ::320].to(device)
            assert speech_inputs_embeds.size(-1) == speech_attention_mask.size(-1)
            speech_inputs_embeds = speech_inputs_embeds.transpose(1,2) #b,t,d
            speech_inputs_embeds = (speech_inputs_embeds - model.embed_mean.to(speech_inputs_embeds.device)) / model.embed_std.to(speech_inputs_embeds.device)
        
            
            #net_speech_inputs_embeds = model.z_proj(speech_inputs_embeds)
            speech_inputs_embeds = speech_inputs_embeds[:,:75*3,:]
            speech_attention_mask = speech_attention_mask[:,:75*3]
            speech_input_length = speech_inputs_embeds.shape[1]
            
            #new_inputs_embeds = torch.concat([text_inputs_embeds, net_speech_inputs_embeds], dim=1) #bsz, seq_len, hidden_size
            new_attention_mask = torch.concat([attention_mask, speech_attention_mask], dim=1)
            
            
            output_sequences = model.generate(
                input_ids=input_ids,
                inputs_embeds=speech_inputs_embeds,
                attention_mask=new_attention_mask,
                max_length=1500,
                #temperature=args.temperature,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
            )
            
            directory = Path("/data/mazhengrui/SpeechLLaMA/scale_encodec/scale_10k")
            mean_path = directory / "mean.pt"
            std_path = directory / "std.pt"
            mean = torch.load(mean_path).to(torch_dtype).to(device)
            std = torch.load(std_path).to(torch_dtype).to(device)
            
            new_embeds = output_sequences[1] * std + mean
            generated_ids = output_sequences[0][:, text_input_length:]

            new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

            wav_len = (generated_ids.ne(2).sum(dim=-1) + speech_input_length) * 320
            
        

            for i in range(len(wav_len)):
                id = batch["id"][i]
                torchaudio.save(output_path / f"{id}.wav", new_audio_values[i][:,:wav_len[i]].cpu(), SAMPLING_RATE)
        
    return


if __name__ == "__main__":
    main()