import os
from pydub import AudioSegment

def resample_audio(input_folder, output_folder, target_sample_rate=16000):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        
        # 检查是否为音频文件（根据扩展名）
        if os.path.isfile(input_path) and filename.lower().endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a')):
            try:
                # 加载音频文件
                audio = AudioSegment.from_file(input_path)
                
                # 重采样到目标采样率
                resampled_audio = audio.set_frame_rate(target_sample_rate)
                
                # 保存到输出文件夹
                output_path = os.path.join(output_folder, filename)
                resampled_audio.export(output_path, format="wav")  # 可根据需要更改格式
                
                print(f"Processed: {input_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# 使用示例
input_folder = "/data/mazhengrui/SpeechLLaMA/eval_continue/librispeech.score.pho.normalize.ffn_2752.12+12_layers.dropout_0.1.act_dropout_0.1.bsz_128.170000.cfg_1.5.seed_0"  # 替换为实际的输入文件夹路径
output_folder = "/data/mazhengrui/SpeechLLaMA/eval_continue/librispeech.score.pho.normalize.ffn_2752.12+12_layers.dropout_0.1.act_dropout_0.1.bsz_128.170000.cfg_1.5.seed_0.16khz"  # 替换为实际的输出文件夹路径
resample_audio(input_folder, output_folder, target_sample_rate=16000)
