import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch.einops import rearrange


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, context_dim=None, num_heads=4):
        super(Attention, self).__init__()
        self.self_attn = context_dim is None
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim if context_dim is not None else hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.key = nn.Linear(self.context_dim, embed_dim, bias=False)
        self.value = nn.Linear(self.context_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, hidden_dim)

    def forward(self, tokens, t=None, context=None):
        B, T, C = tokens.shape
        H = self.num_heads

        Q = self.query(tokens).view(B, T, H, self.head_dim).transpose(1, 2)

        if self.self_attn:
            K = self.key(tokens).view(B, T, H, self.head_dim).transpose(1, 2)
            V = self.value(tokens).view(B, T, H, self.head_dim).transpose(1, 2)
        else:
            _, context_len, context_C = context.shape
            if context_C != self.context_dim:
                context = nn.Linear(context_C, self.context_dim).to(context.device)(context)
                context_C = self.context_dim

            K = self.key(context).view(B, context_len, H, self.head_dim).transpose(1, 2)
            V = self.value(context).view(B, context_len, H, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_probs, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, context_dim, num_heads, self_attn=False, cross_attn=False):
        """
        Build a transformer block
        :param hidden_dim: 图像的隐藏维度（通道数）
        :param context_dim: 文本的隐藏维度
        :param num_heads: Attention中多头的数量
        :param self_attn: 是否使用自注意力
        :param cross_attn: 是否使用交叉注意力
        """
        super(TransformerBlock, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn

        # Self-attention 自注意力
        self.attn_self = Attention(hidden_dim, hidden_dim, num_heads=num_heads) if self_attn else None

        # Cross-attention 交叉注意力
        self.attn_cross = Attention(hidden_dim, hidden_dim, context_dim=context_dim, num_heads=num_heads) if cross_attn else None

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(hidden_dim)

        self.ffn1 = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x, t=None, context=None):
        if self.self_attn:
            x = self.attn_self(self.norm1(x)) + x
            x = self.ffn1(self.norm2(x)) + x

        if self.cross_attn:
            x = self.attn_cross(self.norm3(x), context=context) + x
            x = self.ffn2(self.norm4(x)) + x

        return x


class SpatialTransformer(nn.Module):
    def __init__(self, hidden_dim, context_dim=512, num_heads=4, self_attn=False, cross_attn=False):
        super(SpatialTransformer, self).__init__()
        self.transformer = TransformerBlock(hidden_dim, context_dim, num_heads, self_attn, cross_attn)
        self.context_proj = nn.Linear(context_dim, hidden_dim) if context_dim != hidden_dim else nn.Identity()
        self.self_attn = self_attn
        self.cross_attn = cross_attn

    def forward(self, x, t=None, context=None):
        b, c, h, w = x.shape
        x_res = x  # 用作残差连接
        x = rearrange(x, "b c h w -> b (h w) c")

        if context is not None:
            context = self.context_proj(context)

        x = self.transformer(x, t, context)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x + x_res


class ResnetBlock(nn.Module):
    """
    抄自 Stable Diffusion 1.x.
    源代码中有两个版本的实现：
    1) https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/model.py#L82
    2) https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py#L163
    """
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, in_channels, eps=1e-6)  # SD1.x uses eps=1e-6
        self.norm2 = nn.GroupNorm(4, out_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.SiLU()
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.time_proj = torch.nn.Linear(time_dim, out_channels)

    def forward(self, x, t):
        residual = self.residual_conv(x)

        x = self.conv1(self.activation(self.norm1(x)))
        x = x + self.time_proj(self.activation(t))[:, :, None, None]  # 添加时间嵌入
        x = self.dropout(x)
        x = self.conv2(self.activation(self.norm2(x)))

        return x + residual


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, self_attn=False, cross_attn=False, num_heads=1, context_dim=512):
        super().__init__()
        self.resnet1 = ResnetBlock(in_channels, out_channels, time_dim)
        self.transformer1 = SpatialTransformer(out_channels, context_dim, num_heads=num_heads, self_attn=True) if self_attn else None
        self.resnet2 = ResnetBlock(out_channels, out_channels, time_dim)
        self.transformer2 = SpatialTransformer(out_channels, context_dim, num_heads=num_heads, cross_attn=True) if cross_attn else None
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)  # 下采样

    def forward(self, x, t, y):
        x = self.resnet1(x, t)
        if self.transformer1:
            x = self.transformer1(x, t, y)
        x = self.resnet2(x, t)
        if self.transformer2:
            x = self.transformer2(x, t, y)
        x = self.downsample(x)
        return x


class MiddleBlock(nn.Module):
    def __init__(self, channels, time_dim, context_dim):
        super().__init__()
        self.resnet1 = ResnetBlock(channels, channels, time_dim)
        self.attn1 = SpatialTransformer(channels, context_dim, num_heads=channels//64, self_attn=True, cross_attn=True)  # 256/64=4
        self.resnet2 = ResnetBlock(channels, channels, time_dim)
        # 可选：添加第二个注意力层和resnet块
        self.attn2 = SpatialTransformer(channels, context_dim, num_heads=channels//64, self_attn=True, cross_attn=True)  # 256/64=4
        self.resnet3 = ResnetBlock(channels, channels, time_dim)

    def forward(self, x, t, context):
        x = self.resnet1(x, t)
        x = self.attn1(x, t, context)
        x = self.resnet2(x, t)
        x = self.attn2(x, t, context)
        x = self.resnet3(x, t)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, self_attn=False, cross_attn=False, num_heads=1, context_dim=512):
        super().__init__()
        # nn.Upsample(scale_factor=2, mode='nearest'),
        # nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=1, padding=1),  # or, kernel_size=5, padding=2
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)  # 上采样
        self.resnet1 = ResnetBlock(out_channels, out_channels, time_dim)
        self.transformer1 = SpatialTransformer(out_channels, context_dim, num_heads=num_heads, self_attn=True) if self_attn else None
        self.resnet2 = ResnetBlock(out_channels, out_channels, time_dim)
        self.transformer2 = SpatialTransformer(out_channels, context_dim, num_heads=num_heads, cross_attn=True) if cross_attn else None
        self.resnet3 = ResnetBlock(out_channels, out_channels, time_dim)

    def forward(self, x, t, y):
        x = self.upsample(x)
        x = self.resnet1(x, t)
        if self.transformer1:
            x = self.transformer1(x, t, y)
        x = self.resnet2(x, t)
        if self.transformer2:
            x = self.transformer2(x, t, y)
        x = self.resnet3(x, t)
        return x


class UNet_Transformer(nn.Module):
    def __init__(self, in_channels=3, time_dim=256, context_dim=512):
        super().__init__()

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.context_dim = context_dim

        # 初始卷积
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1)
        )  # 64 x H x W

        # 下采样
        self.down1 = self._down_block(64, 128, time_dim, self_attn=True, cross_attn=False, num_heads=4, context_dim=context_dim)  # 128 x H/2 x W/2
        self.down2 = self._down_block(128, 256, time_dim, self_attn=True, cross_attn=False, num_heads=4, context_dim=context_dim)  # 256 x H/4 x W/4
        self.down3 = self._down_block(256, 512, time_dim, self_attn=True, cross_attn=False, num_heads=8, context_dim=context_dim)  # 512 x H/8 x W/8

        # 中间块
        self.middle_block = MiddleBlock(512, time_dim, context_dim)  # 512 x H/8 x W/8

        # 上采样
        self.up1 = self._up_conv(512, 256, time_dim, self_attn=True, cross_attn=True, num_heads=8, context_dim=context_dim)  # 256 x H/4 x W/4
        self.up2 = self._up_conv(256+256, 128, time_dim, self_attn=True, cross_attn=True, num_heads=4, context_dim=context_dim)  # 128 x H/2 x W/2
        self.up3 = self._up_conv(128+128, 64, time_dim, self_attn=True, cross_attn=True, num_heads=4, context_dim=context_dim)  # 64 x H x W

        # 最终卷积
        self.final_conv = nn.Sequential(
            ResnetBlock(64 * 2, 64, time_dim),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, in_channels, 3, stride=1, padding=1),
        )

    def get_sinusoidal_position_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:  # zero pad if embedding_dim is odd
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def _down_block(self, in_channels, out_channels, time_dim, self_attn=False, cross_attn=False, num_heads=1, context_dim=None):
        return DownBlock(in_channels, out_channels, time_dim, self_attn, cross_attn, num_heads, context_dim or self.context_dim)

    def _up_conv(self, in_channels, out_channels, time_dim, self_attn=False, cross_attn=False, num_heads=1, context_dim=None):
        return UpBlock(in_channels, out_channels, time_dim, self_attn, cross_attn, num_heads, context_dim or self.context_dim)

    def forward(self, x, t, y):
        # x: [batch, 3, H, W]
        # t: [batch, ] time embedding
        # y: [batch, 512] text embedding
        initial_x = x
        # Ensure y has the correct shape
        if y.dim() == 2:
            y = y.unsqueeze(1)  # [batch, 1, context_dim]

        t = self.get_sinusoidal_position_embedding(t, self.time_dim)  # [batch, 256]
        t = self.time_mlp(t)

        x1 = self.init_conv(x)

        x2 = self.down1(x1, t, y)
        x3 = self.down2(x2, t, y)
        x4 = self.down3(x3, t, y)

        x4 = self.middle_block(x4, t, y)

        x = self.up1(x4, t, y)
        x = torch.cat([x, x3], dim=1)  # skip connection 跳跃连接
        x = self.up2(x, t, y)
        x = torch.cat([x, x2], dim=1)  # skip connection 跳跃连接
        x = self.up3(x, t, y)
        x = torch.cat([x, x1], dim=1)  # skip connection 跳跃连接

        x = self.final_conv[0](x, t)
        for layer in self.final_conv[1:]:
            x = layer(x)

        return x + initial_x  # 全局残差连接


"""添加噪声过程：从原始图像开始，逐渐增加噪声，直到最终的噪声图像"""
class NoiseScheduler:
    def __init__(self, num_timesteps, device):
        self.device = device
        self.num_timesteps = num_timesteps
        self.betas = self.cosine_beta_schedule(num_timesteps).to(device) # 这里我们使用余弦噪声调度。DDPM原始论文中使用的是线性调度。
        self.alphas = (1. - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod).to(device)

    def cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(self, x_start, t):
        """
        添加噪声到输入图像或潜在表示。
        :param x_start: 初始清晰图像或潜在表示
        :param t: 当前时间步
        :return: 添加噪声后的表示
        """
        t = t.clone().detach().long().to(self.sqrt_alphas_cumprod.device)
        # 生成标准正态分布的噪声
        noise = torch.randn_like(x_start)
        # 获取所需的预计算值
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        # 计算第t步、带噪声的图像
        x_t = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

        return x_t, noise



"""CFG采样器（去噪）Classifier Guided Diffusion"""
@torch.no_grad()
def sample_cfg(model, noise_scheduler, n_samples, in_channels, text_embeddings, image_size=64, guidance_scale=3.0):
    """
    从噪声开始，逐渐减小噪声，直到最终的图像。
    :param model: UNet模型
    :param noise_scheduler: 噪声调度器
    :param n_samples: 生成的样本数量
    :param in_channels: 输入图像的通道数
    :param text_embeddings: 文本嵌入
    :param image_size: 图像的大小
    :param guidance_scale: 用于加权噪声预测的比例
    :return: 生成的图像
    """
    model.eval()
    device = next(model.parameters()).device

    x = torch.randn(n_samples, in_channels, image_size, image_size).to(device) # 随机初始化噪声图像
    null_embeddings = torch.zeros_like(text_embeddings) # 用于无条件生成

    # 逐步去噪
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)

        noise_pred_uncond = model(x, t_batch, y=null_embeddings) # 生成一个无条件的噪声预测
        noise_pred_cond = model(x, t_batch, y=text_embeddings) # 生成一个有条件的噪声预测
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond) # CFG：结果加权后的噪声预测

        # 采样器的去噪过程
        alpha_t = noise_scheduler.alphas[t]
        alpha_t_bar = noise_scheduler.alphas_cumprod[t]
        beta_t = noise_scheduler.betas[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        # 去噪公式
        x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_bar))) * noise_pred) + torch.sqrt(beta_t) * noise

    model.train()
    return x


"""普通采样器（去噪）"""
@torch.no_grad()
def sample(model, x_t, noise_scheduler, t, text_embeddings):
    """
    从噪声开始，逐渐减小噪声，直到最终的图像。

    参数:
    - model: UNet模型用于预测噪声。
    - x_t: 当前时间步的噪声化表示（torch.Tensor）。
    - noise_scheduler: 噪声调度器，包含betas和其他预计算值。
    - t: 当前时间步（torch.Tensor）。
    - text_embeddings: 文本嵌入，用于条件生成（torch.Tensor）。

    返回:
    - x: 去噪后的表示。
    """
    t = t.to(x_t.device)

    # 获取当前时间步的beta和alpha值
    beta_t = noise_scheduler.betas[t]
    alpha_t = noise_scheduler.alphas[t]
    alpha_t_bar = noise_scheduler.alphas_cumprod[t]

    # 预测当前时间步的噪声
    predicted_noise = model(x_t, t, text_embeddings)

    # 计算去噪后的表示
    if t > 0:
        noise = torch.randn_like(x_t).to(x_t.device)
    else:
        noise = torch.zeros_like(x_t).to(x_t.device)

    # 去噪公式
    x = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_bar))) * predicted_noise) + torch.sqrt(beta_t) * noise

    return x


class DDIMSampler:
    def __init__(self, model, n_steps=50, device="cuda"):
        self.model = model
        self.n_steps = n_steps
        self.device = device

    @torch.no_grad()
    def sample(self, noise, context, guidance_scale=3.0):
        # Assuming your noise scheduler is accessible via model.noise_scheduler
        scheduler = self.model.noise_scheduler

        # Initialize x_t with pure noise
        x = noise

        for i in reversed(range(0, scheduler.num_timesteps, scheduler.num_timesteps // self.n_steps)):
            t = torch.full((noise.shape[0],), i, device=self.device, dtype=torch.long)

            # For classifier-free guidance
            noise_pred_uncond = self.model.unet(x, t, y=torch.zeros_like(context))
            noise_pred_cond = self.model.unet(x, t, y=context)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # DDIM update step
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = scheduler.alphas_cumprod[t-1] if i > 0 else torch.ones_like(alpha_prod_t)

            pred_x0 = (x - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
            dir_xt = torch.sqrt(1 - alpha_prod_t_prev - scheduler.betas[t]) * noise_pred
            x = torch.sqrt(alpha_prod_t_prev) * pred_x0 + dir_xt

        return x