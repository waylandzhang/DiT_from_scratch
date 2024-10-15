import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from datasets import load_dataset
from diffusion_model import UNet_Transformer, NoiseScheduler, sample_cfg
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
import wandb

# 超参数
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
image_size = 64
in_channels = 3
n_epochs = 1000
batch_size = 32
lr = 1e-4
num_timesteps = 1000
save_checkpoint_interval = 100

# WandB 初始化
run = wandb.init(
    project="diffusion_from_scratch",
    config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": n_epochs,
        "image_size": image_size,
        "in_channels": in_channels,
        "num_timesteps": num_timesteps,
    },
)

# 初始化模型和噪声调度器
diffusion_model = UNet_Transformer(in_channels=in_channels).to(device)
noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps, device=device)

# 加载 CLIP 模型
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# WandB 监控
wandb.watch(diffusion_model, log_freq=100)

# 加载数据集
dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")

# 数据预处理
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images, "text": examples["en_text"]}

dataset.set_transform(transform)

train_dataset = dataset.select(range(0, 600)) # 选择前 600 个样本作为训练集
val_dataset = dataset.select(range(600, 800)) # 选择接下来的 200 个样本作为验证集

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 优化器和学习率调度器
optimizer = AdamW(diffusion_model.parameters(), lr=lr, weight_decay=1e-4)  # 可以考虑加入L2正则化：weight_decay=1e-4
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=5e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs) # 余弦退火学习率调度器
# scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=n_epochs, steps_per_epoch=len(train_dataloader)) # OneCycleLR 学习率调度器

# 创建保存生成测试图像的目录
os.makedirs('diffusion_results', exist_ok=True)

# 训练循环
for epoch in range(n_epochs):
    diffusion_model.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{n_epochs}")
    epoch_loss = 0.0

    # 训练模型
    for batch in train_dataloader:
        images = batch["images"].to(device)
        text = batch["text"]

        # 使用 CLIP 模型编码文本
        text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

        timesteps = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long() # 随机选择 timesteps
        noisy_images, noise = noise_scheduler.add_noise(images, timesteps) # 添加噪声
        noise_pred = diffusion_model(noisy_images, timesteps, text_embeddings) # 预测噪声
        loss = torch.nn.functional.mse_loss(noise_pred, noise) # 预测的噪声与真实噪声的均方误差

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        # scheduler.step()  # OneCycleLR 在每个批次后调用

        epoch_loss += loss.item()
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": loss.item()})

    # 验证集上评估模型
    diffusion_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            images = batch["images"].to(device)
            text = batch["text"]
            text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

            timesteps = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()
            noisy_images, noise = noise_scheduler.add_noise(images, timesteps)
            noise_pred = diffusion_model(noisy_images, timesteps, text_embeddings)
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            val_loss += loss.item()

    scheduler.step() # 除了 OneCycleLR 之外，其他调度器都需要在每个 epoch 结束时调用

    wandb.log({
        "epoch": epoch,
        "train_loss": epoch_loss / len(train_dataloader),
        "val_loss": val_loss / len(val_dataloader),
        "learning_rate": scheduler.get_last_lr()[0]
    })

    # 保存模型检查点
    if (epoch + 1) % save_checkpoint_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
        }, f'diffusion_results/diffusion_model_checkpoint_epoch_{epoch+1}.pth')

    # 生成测试图像
    if (epoch + 1) % save_checkpoint_interval == 0:
        diffusion_model.eval()
        with torch.no_grad():
            sample_text = ["a water type pokemon", "a red pokemon with a red fire tail"]
            text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state
            sampled_images = sample_cfg(diffusion_model, noise_scheduler, len(sample_text), in_channels, text_embeddings, image_size=image_size, guidance_scale=3.0)
            # 保存生成的图像
            for i, img in enumerate(sampled_images):
                img = img * 0.5 + 0.5  # Rescale to [0, 1]
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_pil.save(f'diffusion_results/generated_image_epoch_{epoch+1}_sample_{i}.png')

            wandb.log({f"generated_image_{i}": wandb.Image(sampled_images[i]) for i in range(len(sample_text))})

torch.save(diffusion_model.state_dict(), 'diffusion_model_final.pth')
wandb.finish()