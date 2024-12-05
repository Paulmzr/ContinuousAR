import argparse
import logging
from typing import Tuple
from pathlib import Path

import torch
from accelerate.utils import set_seed

from transformers import AutoTokenizer, RobertaTokenizer
from speechllama.speech_llama_score import SpeechLlamaForCausalLM

import pdb

import torchaudio
from datasets import load_dataset, concatenate_datasets, Audio




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
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
    model.infer_cfg = args.cfg
    model.initialize_codec("facebook/encodec_24khz") 
    model.to(device)
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="test.clean").select_columns(["text_normalized", "id"])
    batch_size = args.batch_size
    
    pathname = args.output + f".{args.ckpt}.cfg_{args.cfg}.seed_{args.seed}"
    output_path = Path("eval") / pathname
    output_path.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad(): 
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset.select(range(i, min(i + batch_size, len(eval_dataset))))
            input_text = batch["text_normalized"]
            
            batch_encoded = tokenizer.batch_encode_plus(
                input_text,
                add_special_tokens=True,
                padding="longest",  # 按最长的文本自动填充
                truncation=True,    # 如果文本过长，自动截断
                return_tensors="pt" # 返回 PyTorch 张量（可选）
            )
            #input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
            input_ids = batch_encoded["input_ids"].to(device)
            attention_mask = batch_encoded["attention_mask"].to(device)
            input_length = input_ids.shape[1]

            
            output_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
            generated_ids = output_sequences[0][:, input_length:]

            new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

            wav_len = generated_ids.ne(2).sum(dim=-1) * 320
            
        

            for i in range(len(wav_len)):
                id = batch["id"][i]
                torchaudio.save(output_path / f"{id}.wav", new_audio_values[i][:,:wav_len[i]].cpu(), SAMPLING_RATE)
        
    return


if __name__ == "__main__":
    main()