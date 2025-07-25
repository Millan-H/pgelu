import torch
import torch.nn as nn
import torch.nn.functional as F

class ParametricGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(1, out_channels)
        self.alphas = nn.Parameter(torch.zeros(out_channels))
        self.betas = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
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