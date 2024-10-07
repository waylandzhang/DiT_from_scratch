"""
这段代码用于展示如何使用训练好的VAE模型对图像进行编码和解码。
用自己训练好的vae模型来压缩一张图片（pokemon_sample_test.png）到潜在空间，然后再还原到像素空间并可视化的过程。
需要通过train_vae.py训练好VAE模型并保存后，才能运行这段代码。
"""
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from vae_model import VAE


# 超参数
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
image_size = 512
latent_dim = 4

# 加载一个随机的原始图像
image_path = "pokemon_sample_test.png"
original_image = Image.open(image_path)

preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # 图片大小调整为 512 x 512
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将像素值从 [0, 1] 转换到 [-1, 1]
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}

# 处理图片到3通道的RGB格式（防止有时图片是RGBA的4通道）
image_tensor = preprocess(original_image.convert("RGB")).unsqueeze(0).to(device)

mean_value = image_tensor.mean().item()
print(f"Mean value of image_tensor: {mean_value}")

# 加载我们刚刚预训练好的VAE模型
vae = VAE(in_channels=3, latent_dim=latent_dim, image_size=image_size).to(device)
vae.load_state_dict(torch.load('vae_model.pth', map_location=torch.device('cpu')))

# 使用VAE的encoder压缩图像到潜在空间
with torch.no_grad():
    mu, log_var = vae.encode(image_tensor)
    latent = vae.reparameterize(mu, log_var)

# 使用encoder的输出通过decoder重构图像
with torch.no_grad():
    reconstructed_image = vae.decode(latent)

# 显示原始图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original Image")
plt.axis('off')

# 显示重构图像
reconstructed_image = reconstructed_image.squeeze().cpu().numpy().transpose(1, 2, 0)
reconstructed_image = (reconstructed_image + 1) / 2  # 从[-1, 1]转换到[0, 1]
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title("Reconstructed Image")
plt.axis('off')

plt.show()

# 将潜在向量转换为可视化的图像格式
latent_image = latent.squeeze().cpu().numpy()

# 检查潜在向量的形状
if latent_image.ndim == 1:
    # 如果是1D的，将其reshape成2D图像
    side_length = int(np.ceil(np.sqrt(latent_image.size)))
    latent_image = np.pad(latent_image, (0, side_length**2 - latent_image.size), mode='constant')
    latent_image = latent_image.reshape((side_length, side_length))
elif latent_image.ndim == 3:
    # 如果是3D的，选择一个切片或进行平均
    latent_image = np.mean(latent_image, axis=0)

# 显示潜在向量图像
plt.imshow(latent_image, cmap='gray')
plt.title("Latent Space Image")
plt.axis('off')
plt.colorbar()
plt.show()
