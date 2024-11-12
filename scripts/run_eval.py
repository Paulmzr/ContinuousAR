import argparse
import logging
from typing import Tuple

import torch
from accelerate.utils import set_seed

from transformers import AutoProcessor, AutoTokenizer
from speechllama.speech_llama import SpeechLlamaForCausalLM, ModifiedGenerateDecoderOnlyOutput
from dac.model.feature_extraction_vae import DacFeatureExtractor

import pdb

import torchaudio
from datasets import load_dataset, concatenate_datasets, Audio
from train_vae import DataCollatorForSupervisedDataset




SAMPLING_RATE=16000

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

    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--max_length", type=int, default=0)

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch_dtype)
    model.initialize_vae_codec("/data/mazhengrui/codec/descript-audio-codec/runs/vae-2/200k/vae/weights.pth")
    model.to(device)
    model.initialize_scale("/data/mazhengrui/SpeechLLaMA/vae_scale/scale_10k")
    
    assert tokenizer.pad_token is not None
    logger.info(f"tokenizer pad token: {tokenizer.pad_token}")


    max_seq_length = getattr(model.config, "max_position_embeddings", 0)
    args.max_length = adjust_length_to_model(args.max_length, max_sequence_length=max_seq_length)
    logger.info(args)

    
    
    eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="dev.clean")
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    remove_column_names = list(eval_dataset.features)
    remove_column_names.remove("audio")
    
    
    tokenized_eval_dataset = eval_dataset.map(
        lambda example: tokenizer(example["text_normalized"]),
        batched=True,
        remove_columns=remove_column_names,
        load_from_cache_file=True,
    )
    processor = DacFeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        hop_length=320,
        padding_side="right",
        return_attention_mask=True,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, processor=processor)
    
    batch = data_collator([tokenized_eval_dataset[0]])

    #pdb.set_trace()
    batch["input_ids"] = batch["input_ids"].to(device)
    batch["attention_mask"] = batch["attention_mask"].to(device)
    batch["audio_inputs"]["padding_mask"] = batch["audio_inputs"]["padding_mask"].to(device)
    batch["audio_inputs"]["input_values"] = batch["audio_inputs"]["input_values"].to(device)
    
    output = model(**batch)
    
    new_embeds = output.loss.transpose(-1,-2)
    

    new_audio_values = model.codec.decode(new_embeds)
    torchaudio.save("new_output1.wav", new_audio_values[0].cpu(), SAMPLING_RATE)
    
    input_text = args.input if args.input else input("Model input >>> ")    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    
    
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=1500,
        #temperature=args.temperature,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )
    
    
    new_embeds = output_sequences[1].transpose(-1,-2)
    new_audio_values = model.codec.decode(new_embeds)
    torchaudio.save("new_output1.wav", new_audio_values[0].cpu(), SAMPLING_RATE)
    # 


    return


if __name__ == "__main__":
    main()
    
    
    
'''
    duration = 1  # in seconds
    sample_rate = 24000  # in Hz
    total_samples = sample_rate * duration

    # Generate random audio data in the range of typical audio samples (-1 to 1 for normalized audio data)
    import numpy as np
    audio_data = np.random.uniform(low=-1.0, high=1.0, size=total_samples)
    
    inputs = processor(raw_audio=audio_data, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    inputs["padding_mask"] = inputs["padding_mask"].to(device)
    inputs["input_values"] = inputs["input_values"].to(device)
    
    encoder_outputs = model.codec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=6)

'''