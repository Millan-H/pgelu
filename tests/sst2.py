#!/usr/bin/env python3
"""
SST2 Activation Function Comparison Test
Compares ParametricGELU vs standard GELU on SST2 sentiment classification
Both models initialized with identical weights for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import copy
import os

# Set seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

class SimpleTokenizer:
    """Simple tokenizer that builds vocabulary from the dataset"""
    def __init__(self):
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.vocab_size = 2
    
    def build_vocab(self, sentences, max_vocab_size=10000):
        """Build vocabulary from sentences"""
        word_counts = {}
        
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, count in sorted_words[:max_vocab_size-2]:  # -2 for PAD and UNK
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_size = len(self.word_to_idx)
        print(f"Built vocabulary with {self.vocab_size} words")
    
    def encode(self, sentence, max_length=128):
        """Convert sentence to token IDs"""
        words = sentence.lower().split()
        token_ids = []
        
        for word in words:
            token_ids.append(self.word_to_idx.get(word, 1))  # 1 is <UNK>
        
        # Pad or truncate
        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        else:
            token_ids.extend([0] * (max_length - len(token_ids)))  # 0 is <PAD>
        
        # Create attention mask
        attention_mask = [1 if token_id != 0 else 0 for token_id in token_ids]
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long)
        }

class SST2Dataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        encoding = self.tokenizer.encode(sentence, self.max_length)
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

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

def load_sst2_data():
    """Load SST2 dataset from the specified directory"""
    data_dir = "C:/Users/milla/spikeactivations/sst2_dataset/data"
    
    print(f"Loading SST2 dataset from: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load the parquet files
    train_file = os.path.join(data_dir, "train-00000-of-00001.parquet")
    val_file = os.path.join(data_dir, "validation-00000-of-00001.parquet")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(val_file):
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    
    print("Loading parquet files...")
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)
    
    # Use a subset for faster training (remove this line for full dataset)
    train_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
    
    print(f"Train examples: {len(train_df)}")
    print(f"Validation examples: {len(val_df)}")
    print(f"Sample training data:")
    print(train_df.head(3))
    
    return train_df, val_df

def initialize_models_with_same_weights(vocab_size, embed_dim, hidden_dim, num_classes):
    """Initialize both models with identical weights"""
    
    # Create both activation layers
    parametric_gelu_layer = ParametricGELU(embed_dim, hidden_dim)
    standard_gelu_layer = StandardGELU(embed_dim, hidden_dim)
    
    # Create both models
    model_parametric = SimpleClassifier(vocab_size, embed_dim, hidden_dim, num_classes, parametric_gelu_layer)
    model_standard = SimpleClassifier(vocab_size, embed_dim, hidden_dim, num_classes, standard_gelu_layer)
    
    # Copy weights from parametric to standard (excluding parametric-specific parameters)
    with torch.no_grad():
        # Copy embedding weights
        model_standard.embedding.weight.copy_(model_parametric.embedding.weight)
        
        # Copy linear and layernorm weights from fc1
        model_standard.fc1.linear.weight.copy_(model_parametric.fc1.linear.weight)
        model_standard.fc1.linear.bias.copy_(model_parametric.fc1.linear.bias)
        model_standard.fc1.layernorm.weight.copy_(model_parametric.fc1.layernorm.weight)
        model_standard.fc1.layernorm.bias.copy_(model_parametric.fc1.layernorm.bias)
        
        # Copy fc2 weights
        model_standard.fc2.weight.copy_(model_parametric.fc2.weight)
        model_standard.fc2.bias.copy_(model_parametric.fc2.bias)
    
    return model_parametric, model_standard

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f"    Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    
    return avg_loss, accuracy

def plot_training_curves(parametric_losses, standard_losses, parametric_accs, standard_accs):
    """Plot training curves for comparison"""
    epochs = range(1, len(parametric_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, parametric_losses, 'b-', label='ParametricGELU', marker='o')
    ax1.plot(epochs, standard_losses, 'r-', label='StandardGELU', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, parametric_accs, 'b-', label='ParametricGELU', marker='o')
    ax2.plot(epochs, standard_accs, 'r-', label='StandardGELU', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('sst2_activation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load dataset
        train_df, val_df = load_sst2_data()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the SST2 dataset is downloaded to the specified directory.")
        return
    
    # Initialize simple tokenizer
    print("Building vocabulary...")
    tokenizer = SimpleTokenizer()
    
    # Build vocabulary from training data
    all_sentences = train_df['sentence'].tolist() + val_df['sentence'].tolist()
    tokenizer.build_vocab(all_sentences, max_vocab_size=10000)
    
    # Create datasets
    train_dataset = SST2Dataset(
        train_df['sentence'].tolist(), 
        train_df['label'].tolist(), 
        tokenizer
    )
    val_dataset = SST2Dataset(
        val_df['sentence'].tolist(), 
        val_df['label'].tolist(), 
        tokenizer
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Model parameters
    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    hidden_dim = 256
    num_classes = 2
    
    # Initialize models with same weights
    print("Initializing models with identical weights...")
    model_parametric, model_standard = initialize_models_with_same_weights(
        vocab_size, embed_dim, hidden_dim, num_classes
    )
    
    # Move models to device
    model_parametric.to(device)
    model_standard.to(device)
    
    # Print model info
    print(f"\nModel Parameters:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # Count parameters
    param_count = sum(p.numel() for p in model_parametric.parameters())
    print(f"Total parameters: {param_count:,}")
    
    # Setup optimizers (same learning rate and parameters)
    lr = 2e-4
    optimizer_parametric = optim.Adam(model_parametric.parameters(), lr=lr)
    optimizer_standard = optim.Adam(model_standard.parameters(), lr=lr)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    num_epochs = 10
    
    # Storage for results
    parametric_train_losses = []
    parametric_val_losses = []
    parametric_val_accs = []
    
    standard_train_losses = []
    standard_val_losses = []
    standard_val_accs = []
    
    print(f"\n{'='*80}")
    print("STARTING TRAINING COMPARISON")
    print(f"{'='*80}")
    print(f"{'Epoch':<6} {'Model':<15} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<12}")
    print(f"{'-'*80}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Train ParametricGELU model
        print("Training ParametricGELU model...")
        train_loss_p, train_acc_p = train_epoch(
            model_parametric, train_loader, optimizer_parametric, criterion, device
        )
        val_loss_p, val_acc_p = evaluate(model_parametric, val_loader, criterion, device)
        
        # Train Standard GELU model  
        print("Training StandardGELU model...")
        train_loss_s, train_acc_s = train_epoch(
            model_standard, train_loader, optimizer_standard, criterion, device
        )
        val_loss_s, val_acc_s = evaluate(model_standard, val_loader, criterion, device)
        
        # Store results
        parametric_train_losses.append(train_loss_p)
        parametric_val_losses.append(val_loss_p)
        parametric_val_accs.append(val_acc_p)
        
        standard_train_losses.append(train_loss_s)
        standard_val_losses.append(val_loss_s)
        standard_val_accs.append(val_acc_s)
        
        # Print results
        print(f"{epoch+1:<6} {'ParametricGELU':<15} {train_loss_p:<12.4f} {val_loss_p:<12.4f} {val_acc_p:<12.4f}")
        print(f"{epoch+1:<6} {'StandardGELU':<15} {train_loss_s:<12.4f} {val_loss_s:<12.4f} {val_acc_s:<12.4f}")
        print()
    
    # Final comparison
    print(f"{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    
    final_parametric_acc = parametric_val_accs[-1]
    final_standard_acc = standard_val_accs[-1]
    
    print(f"ParametricGELU Final Validation Accuracy: {final_parametric_acc:.4f}")
    print(f"StandardGELU Final Validation Accuracy:   {final_standard_acc:.4f}")
    print(f"Improvement: {final_parametric_acc - final_standard_acc:+.4f}")
    
    if final_parametric_acc > final_standard_acc:
        improvement_pct = ((final_parametric_acc - final_standard_acc) / final_standard_acc) * 100
        print(f"ParametricGELU is {improvement_pct:.2f}% better than StandardGELU")
    else:
        decline_pct = ((final_standard_acc - final_parametric_acc) / final_standard_acc) * 100
        print(f"ParametricGELU is {decline_pct:.2f}% worse than StandardGELU")
    
    # Plot training curves
    try:
        plot_training_curves(
            parametric_val_losses, standard_val_losses,
            parametric_val_accs, standard_val_accs
        )
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'epoch': range(1, num_epochs + 1),
        'parametric_train_loss': parametric_train_losses,
        'parametric_val_loss': parametric_val_losses,
        'parametric_val_acc': parametric_val_accs,
        'standard_train_loss': standard_train_losses,
        'standard_val_loss': standard_val_losses,
        'standard_val_acc': standard_val_accs
    })
    
    results_df.to_csv('sst2_activation_comparison_results.csv', index=False)
    print(f"\nDetailed results saved to: sst2_activation_comparison_results.csv")
    
    # Analysis of ParametricGELU parameters
    print(f"\n{'='*80}")
    print("PARAMETRICGELU PARAMETER ANALYSIS")
    print(f"{'='*80}")
    with torch.no_grad():
        alphas = model_parametric.fc1.alphas.cpu().numpy()
        betas = model_parametric.fc1.betas.cpu().numpy()
        
        print(f"Alpha parameters - Mean: {alphas.mean():.4f}, Std: {alphas.std():.4f}")
        print(f"Beta parameters - Mean: {betas.mean():.4f}, Std: {betas.std():.4f}")
        print(f"Alpha range: [{alphas.min():.4f}, {alphas.max():.4f}]")
        print(f"Beta range: [{betas.min():.4f}, {betas.max():.4f}]")
        
        # Show some individual values
        print(f"\nFirst 10 alpha values: {alphas[:10]}")
        print(f"First 10 beta values: {betas[:10]}")

if __name__ == "__main__":
    main()