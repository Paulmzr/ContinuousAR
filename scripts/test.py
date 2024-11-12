from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor, AutoTokenizer
import torchaudio

SAMPLING_RATE=24000

eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="dev.clean")
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

model = EncodecModel.from_pretrained("facebook/encodec_24khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
tokenizer = AutoTokenizer.from_pretrained("/data/SharedResources/models/LLM/Llama-2-7b", add_eos_token=True)

examples = eval_dataset[:5]
audio_arrays = [x["array"] for x in examples["audio"]]
audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
text_inputs = tokenizer(examples["text_normalized"])

def preprocess_audio_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt")
    return inputs




# test batch
audio_sample = [eval_dataset[1]["audio"]["array"], eval_dataset[2]["audio"]["array"], eval_dataset[3]["audio"]["array"]] 
inputs = processor(raw_audio=audio_sample, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t

encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=6) #1,b,r,t, 1 due to one chunk
embeds = model.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t
downsampled_mask = inputs["padding_mask"][..., ::320]








audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t
encoder_outputs = codec.encode(audio_inputs["input_values"], audio_inputs["padding_mask"], bandwidth=6) #1,b,r,t, 1 due to one chunk
embeds = codec.quantizer.decode(encoder_outputs.audio_codes[0].transpose(0, 1)) #b,d,t




#index=1
#new_embeds = embeds[index,:,:downsampled_mask[index].sum()].unsqueeze(0)
#new_audio_values = model.decoder(new_embeds)
#torchaudio.save("new_output1.wav", new_audio_values[0], SAMPLING_RATE)


#audio_values = model.decoder(embeds)
#output = audio_values[..., : inputs["padding_mask"].shape[-1]]

#output = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"]).audio_values   #b,c,t

#torchaudio.save("output1.wav", output[0], SAMPLING_RATE)
#torchaudio.save("output2.wav", output[1], SAMPLING_RATE)
#torchaudio.save("output3.wav", output[2], SAMPLING_RATE)


'''
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch
eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=dataset.column_names)
'''
