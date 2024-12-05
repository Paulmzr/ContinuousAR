import pandas as pd
from jiwer import compute_measures, wer

# 读取 TSV 文件
file_path = '/data/mazhengrui/SpeechLLaMA/results/score.bpe.normalize.tiny.12+12_layers.bsz_64.200000.cfg_1.5.seed_42'  # 替换为你的TSV文件路径
df = pd.read_csv(file_path, sep='\t')

reference_text_list = df['normalized_reference_text'].fillna('').tolist()
prediction_text_list = df['normalized_predicted_text'].fillna('').tolist()

incorrect = 0
total = 0
wers = []


for prediction, reference in zip(prediction_text_list, reference_text_list):

    measures = compute_measures(reference, prediction)
    incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
    total += measures["substitutions"] + measures["deletions"] + measures["hits"]
    wers.append(measures['wer'])

concate = compute_measures(" ".join(reference_text_list), " ".join(prediction_text_list))["wer"]
    
print(f"Global: {incorrect / total}")
print(f"Concate: {concate}")
print(f"Average: {sum(wers)/len(wers)}")



