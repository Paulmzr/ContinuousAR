import os
import logging
from datasets import load_dataset, Audio
import whisper
from jiwer import wer, transforms
import argparse
import pandas as pd
from tqdm import tqdm
import csv


SAMPLING_RATE=24000
logger = logging.getLogger(__name__)

def filter_function(example):
    return ((len(example["audio"]["array"]) / 24000) < 10.0) and ((len(example["audio"]["array"]) / 24000) > 4.0)


# 加载 Whisper 模型
model = whisper.load_model("large")  # 选择模型大小: tiny, base, small, medium, large

# 命令行参数解析
parser = argparse.ArgumentParser(description="Evaluate TTS quality using WER.")
parser.add_argument("--audio_folder", type=str, required=True, help="Path to the folder containing TTS-generated audio files.")
parser.add_argument("--output_file", type=str, default="tts_evaluation_results.csv", help="Output file to save evaluation results.")
args = parser.parse_args()

# 配置路径
audio_folder = args.audio_folder
output_file = args.output_file

# 加载数据集
eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="test.clean")
eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    
logger.info(f"original eval dataset: {len(eval_dataset)} samples.")
eval_dataset = eval_dataset.filter(filter_function)
logger.info(f"filtered eval dataset: {len(eval_dataset)} samples.")

# 提取所需列
ids = eval_dataset["id"]
texts = eval_dataset["text_normalized"]

# 确保数据集和音频文件匹配
assert len(ids) == len(texts), "数据集的 ID 列与文本列长度不一致！"

# 定义正则化规则
custom_transform = transforms.Compose([
    transforms.ToLowerCase(),            # 转换为小写
    transforms.RemovePunctuation(),     # 移除标点符号
    transforms.RemoveMultipleSpaces(),       # 移除多余空格
    transforms.Strip(),                      # 去除首尾空格
    transforms.ExpandCommonEnglishContractions(),   
])


if os.path.exists(output_file):
    # 读取已经完成的任务ID列表
    completed_ids = set()
    with open(output_file, mode='r', newline='') as file:
        reader = csv.DictReader(file, delimiter='\t')
        for row in reader:
            completed_ids.add(row['id'])
else:
    completed_ids = set()

# 创建 CSV 文件，添加标题
with open(output_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["id", "normalized_reference_text", "normalized_predicted_text", "wer"], delimiter='\t')
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:  # 只有文件为空时才写入标题
        writer.writeheader()

wers = []
normalized_reference = []
normalized_predictions = []

# 开始评估
for sample_id, reference_text in tqdm(zip(ids, texts), total=len(ids)):
    if sample_id in completed_ids:  # 如果当前ID已完成，则跳过
        continue

    audio_path = os.path.join(audio_folder, f"{sample_id}.wav")
    
    # 检查音频文件是否存在
    if not os.path.exists(audio_path):
        print(f"警告: 找不到音频文件 {audio_path}")
        continue
    
    # 使用 Whisper 转录音频
    transcription = model.transcribe(audio_path)
    predicted_text = transcription["text"].strip()
    
    normalized_predicted_text = custom_transform(predicted_text)
    normalized_reference_text = custom_transform(reference_text)

    # 计算 WER
    error_rate = wer(normalized_reference_text, normalized_predicted_text)
    wers.append(error_rate)
    normalized_reference.append(normalized_reference_text)
    normalized_predictions.append(normalized_predicted_text)
    
    # 将结果写入 CSV 文件
    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "normalized_reference_text", "normalized_predicted_text", "wer"], delimiter='\t')
        writer.writerow({
            "id": sample_id,
            "normalized_reference_text": normalized_reference_text,
            "normalized_predicted_text": normalized_predicted_text,
            "wer": error_rate
        })

# 拼接所有参考文本和预测文本
all_reference_text = " ".join(normalized_reference)
all_predicted_text = " ".join(normalized_predictions)

# 计算全局 WER
global_wer = wer(all_reference_text, all_predicted_text)
print(f"Global WER: {global_wer:.2f}")

# 计算句子级别 WER 的平均值
average_wer = sum(wers) / len(wers) if wers else 0
print(f"Average Sentence-Level WER: {average_wer:.2f}")
