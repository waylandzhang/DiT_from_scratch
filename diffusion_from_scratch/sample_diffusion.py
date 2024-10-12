import torch
from diffusion_model import UNet_Transformer, NoiseScheduler, sample_cfg, sample
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np

# 超参数
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
image_size = 64
in_channels = 3
num_timesteps = 1000

"""从保存点加载"""
checkpoint = torch.load('diffusion_model_checkpoint_epoch_500.pth', map_location=device, weights_only=True)
diffusion_model = UNet_Transformer(in_channels=in_channels).to(device)
diffusion_model.load_state_dict(checkpoint['model_state_dict'])
diffusion_model.eval()

"""从最终模型加载"""
# diffusion_model = UNet_Transformer(in_channels=in_channels).to(device)
# diffusion_model.load_state_dict(torch.load('diffusion_model_final.pth'))
# diffusion_model.eval()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=device)

# 文本条件
# condition = "a red pokemon with a red fire tail"
# condition = "a blue rabbit with a yellow belly"
# condition = "a cartoon pikachu with big eyes and big ears"
condition = "a green bird with a red tail and a black nose"


# 文本编码
text_input = tokenizer([condition], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state

# DDPM采样(CFG)
sampled_images = sample_cfg(diffusion_model, noise_scheduler, n_samples=1, in_channels=in_channels, text_embeddings=text_embeddings, image_size=image_size, guidance_scale=1.0)

# 保存生成的图片
img = sampled_images[0] * 0.5 + 0.5  # 缩放到 [0, 1]
img = img.detach().cpu().permute(1, 2, 0).numpy() # [C, H, W] -> [H, W, C] 调整顺序以适应 PIL 画图
img = (img * 255).astype(np.uint8)
img_pil = Image.fromarray(img)
img_pil.save('generated_image_pokemon_cfg.png')


# DDPM采样(普通)
x_t = torch.randn(1, in_channels, image_size, image_size).to(device)

for t in reversed(range(num_timesteps)):
    t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
    x_t = sample(diffusion_model, x_t, noise_scheduler, t_tensor, text_embeddings)

img = x_t[0] * 0.5 + 0.5  # Rescale to [0, 1]
img = img.detach().cpu().permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
img = (img * 255).astype(np.uint8)
img_pil = Image.fromarray(img)
img_pil.save('generated_image_pokemon.png')