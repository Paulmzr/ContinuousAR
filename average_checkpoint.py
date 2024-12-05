from transformers import AutoModel
import torch
from speechllama.speech_llama_score import SpeechLlamaConfig, SpeechLlamaForCausalLM

# List of checkpoint file paths
checkpoint_paths = [
    "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-160000",
    "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-170000",
    "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-180000",
    "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-190000",
    "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-200000",
]

# Load state_dict from each checkpoint
state_dicts = [SpeechLlamaForCausalLM.from_pretrained(path).state_dict() for path in checkpoint_paths]


averaged_state_dict = state_dicts[0].copy()

for key in averaged_state_dict.keys():
    # Compute the average of weights across all checkpoints
    averaged_state_dict[key] = sum(
        state[key] for state in state_dicts
    ) / len(state_dicts)



# Load the configuration of one of the models
config = SpeechLlamaConfig.from_pretrained(checkpoint_paths[0])

# Initialize a new model instance with the averaged weights
model = SpeechLlamaForCausalLM._from_config(config)
model.load_state_dict(averaged_state_dict)

# Save the averaged model
output_path = "/data/mazhengrui/SpeechLLaMA/experiments/score.bpe.normalize.12+12_layers.bsz_64/checkpoint-average"
model.save_pretrained(output_path)
print(f"Averaged model saved at {output_path}")
