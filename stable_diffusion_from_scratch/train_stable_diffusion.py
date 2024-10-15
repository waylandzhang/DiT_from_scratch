import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms
from datasets import load_dataset
from stable_diffusion_model import load_vae_diffusion_model, StableDiffusion
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import os
import wandb

# 超参数
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
image_size = 512
latent_size = 64 # 潜在表示的宽和高，用于生成图像
n_epochs = 1000
batch_size = 16
lr = 1e-4
num_timesteps = 1000
save_checkpoint_interval = 50
lambda_cons = 0.1  # 一致性损失的权重
max_lambda_cons = 1.0  # 最大一致性损失权重
epochs_to_max_lambda = n_epochs  # 达到最大权重所需的epoch数

# WandB 初始化
run = wandb.init(
    project="stable_diffusion_from_scratch",
    config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": n_epochs,
        "num_timesteps": num_timesteps,
    },
)

class AugmentedLatentDataset(Dataset):
    def __init__(self, original_dataset, model, device, num_augmentations=5):
        self.original_dataset = original_dataset
        self.model = model
        self.device = device
        self.num_augmentations = num_augmentations

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

    def __len__(self):
        return len(self.original_dataset) * self.num_augmentations

    def __getitem__(self, idx):
        original_idx = idx // self.num_augmentations
        original_item = self.original_dataset[original_idx]

        image = original_item["images"]
        text = original_item["text"]

        # Apply augmentation
        augmented_image = self.augment(image)

        # Encode to latent space
        with torch.no_grad():
            latent = self.model.encode(augmented_image.unsqueeze(0).to(self.device))

        return {"latents": latent.squeeze(0).cpu(), "text": text}

# 预加载数据集
dataset = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images, "text": examples["en_text"]}

preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images, "text": examples["en_text"]}

dataset.set_transform(transform)

# 初始化合并的VAE+Diffusion模型
model = StableDiffusion(in_channels=3, latent_dim=4, image_size=512, diffusion_timesteps=1000, device=device)
checkpoint = torch.load('stable_diffusion_results/stable_diffusion_model_checkpoint_epoch_100.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.load_vae('vae_model.pth')
# model = load_vae_diffusion_model('vae_model.pth',
#                                  in_channels=3,
#                                  latent_dim=4,
#                                  image_size=512,
#                                  diffusion_timesteps=1000,
#                                  device=device)
model.to(device)

# Create augmented datasets
train_dataset = AugmentedLatentDataset(dataset.select(range(0, 600)), model, device, num_augmentations=5)
val_dataset = dataset.select(range(600, 800))

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

# 加载 CLIP 模型
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# WandB 监控
wandb.watch(model, log_freq=100)

# 冻结VAE参数
for param in model.vae.parameters():
    param.requires_grad = False
# 确保UNet (diffusion model) 参数可训练
for param in model.unet.parameters():
    param.requires_grad = True

# 优化器和学习率调度器
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
# scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
scheduler = OneCycleLR(optimizer, max_lr=1e-4, epochs=n_epochs, steps_per_epoch=len(train_dataloader)) # OneCycleLR 学习率调度器

# 创建保存生成测试图像的目录
os.makedirs('stable_diffusion_results', exist_ok=True)

# 辅助损失函数：多样性损失
def diversity_loss(latents, use_cosine=False):
    """
    计算多样性损失，可选使用余弦相似度
    """
    batch_size = latents.size(0)
    latents_flat = latents.view(batch_size, -1)

    if use_cosine:
        # 使用余弦相似度
        latents_norm = F.normalize(latents_flat, p=2, dim=1)
        similarity = torch.mm(latents_norm, latents_norm.t())
    else:
        # 使用原始的点积相似度
        similarity = torch.mm(latents_flat, latents_flat.t())

    # 移除对角线上的自相似度
    similarity = similarity - torch.eye(batch_size, device=latents.device)

    return similarity.sum() / (batch_size * (batch_size - 1))

diversity_weight = 0.01  # 多样性损失起始权重

# 训练循环
for epoch in range(n_epochs):
    model.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{n_epochs}")
    epoch_loss = 0.0
    num_batches = 0

    # 更新一致性损失权重
    current_lambda_cons = min(lambda_cons * (epoch + 1) / epochs_to_max_lambda, max_lambda_cons)

    # 训练模型
    for batch in train_dataloader:
        latents = batch["latents"].to(device)
        text = batch["text"]

        # 添加噪声
        timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents, noise = model.noise_scheduler.add_noise(latents, timesteps)

        # 使用 CLIP 模型编码文本
        text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

        # 预测噪声
        noise_pred = model(noisy_latents, timesteps, text_embeddings)
        mse_loss = F.mse_loss(noise_pred, noise)
        div_loss = diversity_loss(noisy_latents, use_cosine=True)

        # 计算去噪后的潜在表示
        alpha_t = model.noise_scheduler.alphas[timesteps][:, None, None, None]
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        predicted_latents = (noisy_latents - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        cons_loss = F.mse_loss(predicted_latents, latents)


        # 组合损失
        total_loss = mse_loss + diversity_weight * div_loss + cons_loss * current_lambda_cons
        epoch_loss += total_loss.item()
        num_batches += 1

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()  # OneCycleLR 学习率调度器

        # 动态调整多样性损失的权重
        if epoch % 10 == 0:
            diversity_weight = min(diversity_weight * 1.05, 0.1)  # 逐渐增加权重，但设置上限

        progress_bar.update(1)
        progress_bar.set_postfix({"loss": epoch_loss / num_batches})

    average_train_loss = epoch_loss / num_batches

    # 验证集上评估模型
    model.eval()
    val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_dataloader:
            data = batch["images"].to(device)
            latents = model.encode(data)
            text = batch["text"]

            # 添加噪声
            timesteps = torch.randint(0, num_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents, noise = model.noise_scheduler.add_noise(latents, timesteps)

            # 使用 CLIP 模型编码文本
            text_inputs = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.to(device)).last_hidden_state

            # 预测噪声
            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            mse_loss = F.mse_loss(noise_pred, noise)

            val_loss += mse_loss.item()
            val_batches += 1

    average_val_loss = val_loss / val_batches

    # scheduler.step()

    wandb.log({
        "epoch": epoch,
        "learning_rate": scheduler.get_last_lr()[0],
        "train_loss": average_train_loss,
        "val_loss": average_val_loss,
    })

    # 保存模型检查点
    if (epoch + 1) % save_checkpoint_interval == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
        }, f'stable_diffusion_results/stable_diffusion_model_checkpoint_epoch_{epoch+1}.pth')

    # 生成测试图像
    if (epoch + 1) % save_checkpoint_interval == 0:
        model.eval()
        with torch.no_grad():
            sample_text = ["a water type pokemon", "a red pokemon with a red fire tail"]
            text_input = tokenizer(sample_text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device)).last_hidden_state

            # 使用模型的sample方法生成图像
            sampled_latents = model.sample(text_embeddings, latent_size=latent_size, batch_size=len(sample_text), guidance_scale=7.5, device=device)

            # 使用VAE解码器将潜在表示解码回像素空间
            sampled_images = model.decode(sampled_latents)

            # 保存生成的图像
            for i, img in enumerate(sampled_images):
                img = img * 0.5 + 0.5  # Rescale to [0, 1]
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                img_pil = Image.fromarray(img)
                img_pil.save(f'stable_diffusion_results/generated_image_epoch_{epoch+1}_sample_{i}.png')

            wandb.log({f"generated_image_{i}": wandb.Image(sampled_images[i]) for i in range(len(sample_text))})

torch.save(model.state_dict(), 'stable_diffusion_model_final.pth')
wandb.finish()