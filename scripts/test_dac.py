from datasets import load_dataset, Audio
from dac.model.feature_extraction_vae import DacFeatureExtractor

import dac
import torchaudio

model_path = '/data/mazhengrui/codec/descript-audio-codec/runs/vae-2/100k/vae/weights.pth'
model = dac.VAE.load(model_path)


import pdb
SAMPLING_RATE=16000

eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="dev.clean")
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))


#model = DacModel.from_pretrained("descript/dac_16khz")
#processor = AutoProcessor.from_pretrained("descript/dac_16khz")
processor = DacFeatureExtractor(
    feature_size=1,
    sampling_rate=SAMPLING_RATE,
    padding_value=0.0,
    hop_length=320,
    padding_side="right",
    return_attention_mask=True,
    )


examples = eval_dataset[-5:]
audio_arrays = [x["array"] for x in examples["audio"]]
audio_inputs = processor(raw_audio=audio_arrays, sampling_rate=SAMPLING_RATE, return_tensors="pt") # 'padding_mask': b,t  'input_values': b,c,t

posterior = model.encode(audio_inputs["input_values"])
z = posterior.sample()
speech_attention_mask = audio_inputs["padding_mask"][..., ::320]

y = model.decode(z)[0].cpu()
torchaudio.save('output.flac', y, SAMPLING_RATE, format='flac')
pdb.set_trace()