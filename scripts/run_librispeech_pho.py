import argparse
import logging
from typing import Tuple
from pathlib import Path

import torch
from accelerate.utils import set_seed

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from speechllama.speech_llama_score import SpeechLlamaForCausalLM
from g2p_en import G2p
import string

import pdb

import torchaudio

g2p = G2p()
ignored = {" ", *string.punctuation}  # 定义需要忽略的符号


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

    parser.add_argument("--input", type=str, default="IT WAS SILENT AND GLOOMY BEING TENANTED SOLELY BY THE CAPTIVE AND LIGHTED BY THE DYING EMBERS OF A FIRE WHICH HAD BEEN USED FOR THE PURPOSED OF COOKERY")
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
    model.to(device)
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    input_text = args.input if args.input else input("Model input >>> ")    
    input_phones = g2p(input_text)  # 使用 g2p 转换文本为音素
    phoneme = " ".join(["_" if p in ignored else p for p in input_phones])  # 转换为音素并去除不需要的符号
    
    
    input_ids = tokenizer.encode(phoneme, return_tensors='pt').to(device)

    
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1500,
        #temperature=args.temperature,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )
    
    
    directory = Path("/data/mazhengrui/SpeechLLaMA/scale_encodec/librispeech/scale_10k")
    mean_path = directory / "mean.pt"
    std_path = directory / "std.pt"
    mean = torch.load(mean_path).to(torch_dtype).to(device)
    std = torch.load(std_path).to(torch_dtype).to(device)
    
    new_embeds = output_sequences[1] * std + mean
    
    #temp = mean.expand(new_embeds.size(0),new_embeds.size(1),-1)
    #print(temp.size())
    #print(new_embeds.size())
    '''
    new_embeds = output_sequences[1]
    
    codes = model.codec.quantizer.encode(new_embeds.transpose(1,2), bandwidth=6)
    new_embeds = model.codec.quantizer.decode(codes).transpose(-1,-2)
    '''
    new_audio_values = model.codec.decoder(new_embeds.transpose(-1,-2))

    
    filename = args.output + f".{args.ckpt}.cfg_{args.cfg}.wav" if args.output else "output.wav"

    output_path = Path("samples") / filename
    
    torchaudio.save(output_path, new_audio_values[0].cpu(), SAMPLING_RATE)

    return


if __name__ == "__main__":
    main()