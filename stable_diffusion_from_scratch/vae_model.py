"""
一个非常简单的变分自编码器（VAE）模型教学，用于训练压缩和解压缩图像于潜在空间（Latent Space）。
Encoder和Decoder都是简单的卷积神经网络。
Encoder用于将图像压缩为潜在空间表示，Decoder用于将潜在空间表示解压缩还原到原始图像。

在这个例子中，我们将3x512x512的图像压缩到4x64x64的特征值，并进一步输出潜在空间表示向量 z。
"""
import torch
import torch.nn as nn

# VAE model
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4, image_size=512):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder
        # 3 x 512 x 512 -> 4 x 64 x 64
        self.encoder = nn.Sequential(
            self._conv_block(in_channels, 64),  # 64 x 256 x 256
            self._conv_block(64, 128),  # 128 x 128 x 128
            self._conv_block(128, 256),  # 256 x 64 x 64
        )

        # Encoder 的潜在空间输出
        self.fc_mu = nn.Conv2d(256, latent_dim, 1)  # 4 x 64 x 64 <- Latent Space
        self.fc_var = nn.Conv2d(256, latent_dim, 1)  # 4 x 64 x 64 <- Latent Space

        # Decoder
        # 4 x 64 x 64 -> 3 x 512 x 512
        self.decoder_input = nn.ConvTranspose2d(latent_dim, 256, 1)  # 256 x 64 x 64
        self.decoder = nn.Sequential(
            self._conv_transpose_block(256, 128),  # 128 x 128 x 128
            self._conv_transpose_block(128, 64),  # 64 x 256 x 256
            self._conv_transpose_block(64, in_channels),  # 3 x 512 x 512
        )

        self.sigmoid = nn.Sigmoid()  # [0, 1]
        self.tanh = nn.Tanh()  # [-1, 1]

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(),
            nn.LeakyReLU(0.2)
        )

    def _conv_transpose_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            # nn.GroupNorm(num_groups=1, num_channels=out_channels),
            nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(),
            nn.LeakyReLU(0.2)
        )

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        result = self.decoder_input(z)
        result = self.decoder(result)
        # result = self.sigmoid(result)  # 如果原始图像被归一化为[0, 1]，则使用sigmoid
        result = self.tanh(result)  # 如果原始图像被归一化为[-1, 1]，则使用tanh
        # return result.view(-1, self.in_channels, self.image_size, self.image_size)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        """
        返回4个值：
        reconstruction, input, mu, log_var
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)  # 潜在空间的向量表达 Latent Vector z
        return self.decode(z), input, mu, log_var