import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # Use GroupNorm as LayerNorm alternative for 2D
        self.norm = nn.GroupNorm(1, out_channels)  # 1 group = LayerNorm-like
        self.alphas = nn.Parameter(torch.zeros(out_channels))
        self.betas = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        # Reshape for broadcasting: (batch, channels, height, width)
        alphas = self.alphas.view(1, -1, 1, 1)
        betas = self.betas.view(1, -1, 1, 1)
        gelu_input = betas * (x - alphas)
        return F.gelu(gelu_input)

class ParametricGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.alphas = nn.Parameter(torch.zeros(output_dim))
        self.betas = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x):
        linear_out = self.linear(x)
        normalized_out = self.layernorm(linear_out)
        gelu_input = self.betas * (normalized_out - self.alphas)
        return F.gelu(gelu_input)

parametricNetwork=nn.Sequential(
    # First convolutional block
    ParametricGELU2d(3, 64),  # 32x32x64
    ParametricGELU2d(64, 64),  # 32x32x64
    nn.MaxPool2d(2, 2),  # 16x16x64

    # Second convolutional block
    ParametricGELU2d(64, 128),  # 16x16x128
    ParametricGELU2d(128, 128),  # 16x16x128
    nn.MaxPool2d(2, 2),  # 8x8x128

    # Third convolutional block
    ParametricGELU2d(128, 256),  # 8x8x256
    ParametricGELU2d(256, 256),  # 8x8x256
    nn.MaxPool2d(2, 2),  # 4x4x256

    # Flatten and fully connected layers
    nn.Flatten(),  # 4*4*256 = 4096
    ParametricGELU(4096, 1024),
    ParametricGELU(1024, 1024),
    nn.Linear(1024, 10)
).to("cuda")

regularNetowrk=nn.Sequential(
    # First convolutional block
    nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x64
    nn.GroupNorm(1, 64),  # Equivalent to LayerNorm for 2D
    nn.GELU(),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 32x32x64
    nn.GroupNorm(1, 64),
    nn.GELU(),
    nn.MaxPool2d(2, 2),  # 16x16x64

    # Second convolutional block
    nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16x128
    nn.GroupNorm(1, 128),
    nn.GELU(),
    nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128
    nn.GroupNorm(1, 128),
    nn.GELU(),
    nn.MaxPool2d(2, 2),  # 8x8x128

    # Third convolutional block
    nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8x8x256
    nn.GroupNorm(1, 256),
    nn.GELU(),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8x256
    nn.GroupNorm(1, 256),
    nn.GELU(),
    nn.MaxPool2d(2, 2),  # 4x4x256

    # Flatten and fully connected layers
    nn.Flatten(),  # 4*4*256 = 4096
    nn.Linear(4096, 1024),
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, 1024),
    nn.LayerNorm(1024),
    nn.GELU(),
    nn.Linear(1024, 10)
).to("cuda")