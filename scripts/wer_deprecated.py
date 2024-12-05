import os
from datasets import load_dataset
import whisper
from jiwer import wer, transforms
import argparse
import pandas as pd
from tqdm import tqdm
import csv

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
eval_dataset = load_dataset("mythicinfinity/libritts", "clean", split="test.clean").select_columns(["text_normalized", "id"])


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

# 如果输出文件存在，先删除
if os.path.exists(output_file):
    os.remove(output_file)

# 创建 CSV 文件，添加标题
with open(output_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["id", "normalized_reference_text", "normalized_predicted_text", "wer"], delimiter='\t')
    writer.writeheader()


wers = []
normalized_reference = []
normalized_predictions = []

# 开始评估
for sample_id, reference_text in tqdm(zip(ids, texts), total=len(ids)):
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

