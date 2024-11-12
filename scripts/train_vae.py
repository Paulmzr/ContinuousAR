# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import math
import logging
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Sequence, List, Union

import torch
import datasets
import transformers


from datasets import load_dataset, concatenate_datasets, Audio


from transformers import (
    HfArgumentParser,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from transformers import AutoProcessor, AutoTokenizer

from speechllama.speech_llama import SpeechLlamaConfig, SpeechLlamaForCausalLM
from speechllama.trainer import SpeechLlamaTrainer
from dac.model.feature_extraction_vae import DacFeatureExtractor

logger = logging.getLogger(__name__)



SAMPLING_RATE=16000
    

@dataclass
class ArchArguments:
    # --------------------------------------------------------------------------
    # Llama Arguments
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    pretraining_tp: int = 1
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[float] = None
    attention_bias: bool = False
    attention_dropout: float = 0.1
    mlp_bias: bool = False
    
    # --------------------------------------------------------------------------
    # Diffusion Arguments
    vae_embed_dim: int = 32
    diffloss_d: int = 3
    diffloss_w: int = 1024
    num_sampling_steps: str = "20"
    diffusion_batch_mul: int = 4
    learn_sigma: bool = True
    sigma_small: bool = False
    predict_xstart: bool = False
    rescale_learned_sigmas: bool = False



@dataclass
class ModelArguments:
    # --------------------------------------------------------------------------
    # Codec & Tokenizer Arguments
    codec: str = "/data/mazhengrui/codec/descript-audio-codec/runs/vae-2/200k/vae/weights.pth"
    tokenizer: str = "/data/SharedResources/models/LLM/Llama-2-7b"
    scale:str = "/data/mazhengrui/SpeechLLaMA/vae_scale/scale_10k"
    
    '''
    # --------------------------------------------------------------------------
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    '''


@dataclass
class DataArguments:
    data_path: str = "mythicinfinity/libritts"
    train_config: str = "all"
    train_split: Union[str, List[str]] = field(default_factory=lambda: ["train.clean.100", "train.clean.360", "train.other.500"])
    eval_config: str = "clean"
    eval_split: Union[str, List[str]] = field(default_factory=lambda: "dev.clean")
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_multiple_of: Optional[int] = None
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    group_by_speech_length: bool = field(default=True)
    #model_max_length: int = field(
    #    default=512,
    #    metadata={
    #        "help":
    #        "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #    },
    #)



@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    processor: transformers.PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids":instance["input_ids"]} for instance in instances]
    
        '''
        batch = self.tokenizer.pad(
            input_ids,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt"
        )
        '''
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            input_ids,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt"
        )
        
        audio_arrays = [instance["audio"]["array"] for instance in instances]
        audio_inputs = self.processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
        
        batch["audio_inputs"] = audio_inputs
        
        #batch["input_values"] = audio_inputs["input_values"] #b,c,t
        #batch["input_values_mask"] = audio_inputs["padding_mask"] #b,t
        
        '''
        encoder_outputs = self.codec.encode(audio_inputs["input_values"], audio_inputs["padding_mask"], bandwidth=6) #1,b,r,t, 1 due to one chunk
        embeds = self.codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
        embeds_mask = audio_inputs["padding_mask"][..., ::320]
        assert embeds.size(-1) == embeds_mask.size(-1)

        
        
        batch["speech_embeds"] = embeds.transpose(1,2)
        batch["speech_mask"] = embeds_mask
        '''
        
        return batch



def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                arch_args, model_args, data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    if training_args.do_train:
        train_dataset = load_dataset(data_args.data_path, data_args.train_config, split=data_args.train_split)
        train_dataset = concatenate_datasets(train_dataset)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    if training_args.do_eval:
        eval_dataset = load_dataset(data_args.data_path, data_args.eval_config, split=data_args.eval_split)
        eval_dataset = concatenate_datasets(eval_dataset)   
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        
        eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
    remove_column_names = list(eval_dataset.features)
    remove_column_names.remove("audio")
    
    tokenized_train_dataset = None
    tokenized_eval_dataset = None
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            if training_args.do_train:
                tokenized_train_dataset = train_dataset.map(
                    lambda example: tokenizer(example["text_normalized"]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=remove_column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
            if training_args.do_eval:
                tokenized_eval_dataset = eval_dataset.map(
                    lambda example: tokenizer(example["text_normalized"]),
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=remove_column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on eval dataset",
                )
        else:
            if training_args.do_train:
                tokenized_train_dataset = train_dataset.map(
                    lambda example: tokenizer(example["text_normalized"]),
                    batched=True,
                    remove_columns=remove_column_names,
                )
            if training_args.do_eval:
                tokenized_eval_dataset = eval_dataset.map(
                    lambda example: tokenizer(example["text_normalized"]),
                    batched=True,
                    remove_columns=remove_column_names,
                )

    def filter_function(example):
        return (len(example['input_ids']) + len(example["audio"]["array"]) / 320) < arch_args.max_position_embeddings
    
    if tokenized_train_dataset is not None:
        logger.info(f"original train dataset: {len(tokenized_train_dataset)} samples.")
        tokenized_train_dataset = tokenized_train_dataset.filter(filter_function)
        logger.info(f"filtered train dataset: {len(tokenized_train_dataset)} samples.")
        
            
    if tokenized_eval_dataset is not None:
        logger.info(f"original eval dataset: {len(tokenized_eval_dataset)} samples.")
        tokenized_eval_dataset = tokenized_eval_dataset.filter(filter_function)
        logger.info(f"filtered eval dataset: {len(tokenized_eval_dataset)} samples.")
    
    #processor = AutoProcessor.from_pretrained(model_args.codec)
    processor = DacFeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLING_RATE,
        padding_value=0.0,
        hop_length=320,
        padding_side="right",
        return_attention_mask=True,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, processor=processor, pad_to_multiple_of=data_args.pad_to_multiple_of)
    
    return tokenized_train_dataset, tokenized_eval_dataset, data_collator


def train(attn_implementation="flash_attention_2"):

    parser = HfArgumentParser(
        (ArchArguments, ModelArguments, DataArguments, TrainingArguments))
    arch_args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer,
        #model_max_length=training_args.model_max_length,
        padding_side="left",
        add_eos_token=True,
    )
    
    arch_args.vocab_size = tokenizer.vocab_size
    model_config = SpeechLlamaConfig(**asdict(arch_args))
    logger.info(f"config: {model_config}")
    
    
    torch_dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    
    model = SpeechLlamaForCausalLM._from_config(model_config, attn_implementation=attn_implementation, torch_dtype=torch_dtype)
    model.initialize_vae_codec(model_args)
    model.initialize_scale(model_args)  
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")


    train_dataset, eval_dataset, data_collator = make_supervised_data_module(tokenizer, arch_args, model_args, data_args, training_args)
    trainer = SpeechLlamaTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
if __name__ == "__main__":
    train()
