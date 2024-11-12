import torch
import matplotlib.pyplot as plt


# 加载并确认保存的 Tensor
loaded_tensor = torch.load('tensor.pt')

# 使用 Matplotlib 进行绘图
plt.plot(loaded_tensor)  # 将 Tensor 转换为 NumPy 数组
plt.title('Tensor Plot')
plt.xlabel('Index')
plt.ylabel('Value')

# 保存图像到文件
plt.savefig('tensor_plot.png')

# 显示图像
plt.show()
