import torch
import torch.nn as nn
import torch.nn.functional as F

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

class StandardGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        linear_out = self.linear(x)
        normalized_out = self.layernorm(linear_out)
        return F.gelu(normalized_out)
    
class SimpleClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_classes=2, activation_layer=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout1 = nn.Dropout(0.3)
        
        if activation_layer is not None:
            self.fc1 = activation_layer
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
        
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # Simple mean pooling over embeddings
        embeddings = self.embedding(input_ids)
        
        # Apply attention mask and average
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask_expanded
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        
        x = self.dropout1(mean_pooled)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x