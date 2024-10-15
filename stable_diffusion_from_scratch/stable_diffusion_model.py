import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_model import UNet_Transformer, NoiseScheduler
from vae_model import VAE

class StableDiffusion(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, image_size=512, diffusion_timesteps=1000, device="cuda"):
        super(StableDiffusion, self).__init__()

        # VAE
        self.vae = VAE(in_channels=in_channels, latent_dim=latent_dim, image_size=image_size)

        # Diffusion model (UNet)
        self.unet = UNet_Transformer(in_channels=latent_dim)

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(num_timesteps=diffusion_timesteps, device=device)

    def encode(self, x):
        return self.vae.encode(x)[0]

    def decode(self, z):
        return self.vae.decode(z)

    def diffuse(self, latents, t, context):
        return self.unet(latents, t, context)

    def forward(self, latents, t, context):

        noise_pred = self.diffuse(latents, t, context)

        return noise_pred

    def sample(self, context, latent_size=64, batch_size=1, guidance_scale=3.0, device="cuda"):
        # Generate initial random noise in the latent space
        latents = torch.randn(batch_size, self.vae.latent_dim, latent_size, latent_size).to(device)

        # Create unconditioned embedding for classifier-free guidance
        uncond_embeddings = torch.zeros_like(context)

        # Gradually denoise the latents
        for t in reversed(range(self.noise_scheduler.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict noise for both conditioned and unconditioned
            noise_pred_uncond = self.diffuse(latents, t_batch, uncond_embeddings)
            noise_pred_cond = self.diffuse(latents, t_batch, context)

            # Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Get alpha and beta valuescpu
            alpha_t = self.noise_scheduler.alphas[t]
            alpha_t_bar = self.noise_scheduler.alphas_cumprod[t]
            beta_t = self.noise_scheduler.betas[t]

            # Compute "previous" noisy sample x_t -> x_t-1
            if t > 0:
                noise = torch.randn_like(latents)
            else:
                noise = torch.zeros_like(latents)

            latents = (1 / torch.sqrt(alpha_t)) * (
                    latents - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_bar))) * noise_pred
            ) + torch.sqrt(beta_t) * noise

        # Return the final latents instead of decoding them
        return latents

    def load_vae(self, vae_path):
        self.vae.load_state_dict(torch.load(vae_path, map_location=torch.device('cpu')))

    def load_diffusion(self, diffusion_path):
        self.unet.load_state_dict(torch.load(diffusion_path))

class DDIMSampler:
    def __init__(self, model, n_steps=50, device="cuda"):
        self.model = model
        self.n_steps = n_steps
        self.device = device

    @torch.no_grad()
    def sample(self, noise, context, guidance_scale=3.0):

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

def load_vae_diffusion_model(vae_path, in_channels=3, latent_dim=4, image_size=512, diffusion_timesteps=1000, device="cuda"):
    model = StableDiffusion(in_channels, latent_dim, image_size, diffusion_timesteps, device=device)
    model.load_vae(vae_path)
    return model

def load_model_from_checkpoint(checkpoint_path, in_channels=3, latent_dim=4, image_size=512, diffusion_timesteps=1000, device="cuda"):
    model = StableDiffusion(in_channels=in_channels, latent_dim=latent_dim, image_size=image_size, diffusion_timesteps=diffusion_timesteps, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from checkpoint at epoch {checkpoint['epoch']}")
    return model