import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import defaultdict

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class ParametricGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        # Use GroupNorm as LayerNorm alternative for 2D
        self.norm = nn.GroupNorm(1, out_channels)
        self.alphas = nn.Parameter(torch.normal(mean=0, std=0.05, size=(out_channels,)))
        self.betas = nn.Parameter(torch.normal(mean=1, std=0.05, size=(out_channels,)))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        # Reshape for broadcasting: (batch, channels, height, width)
        alphas = self.alphas.view(1, -1, 1, 1)
        betas = self.betas.view(1, -1, 1, 1)
        gelu_input = (x - alphas)
        return F.gelu(gelu_input)

class ParametricGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.layernorm = nn.LayerNorm(output_dim)
        self.alphas = nn.Parameter(torch.normal(mean=0, std=0.05, size=(output_dim,)))
        self.betas = nn.Parameter(torch.normal(mean=1, std=0.05, size=(output_dim,)))
        
    def forward(self, x):
        linear_out = self.linear(x)
        normalized_out = self.layernorm(linear_out)
        gelu_input = (normalized_out - self.alphas)
        return F.gelu(gelu_input)

def load_cifar_data():
    """Load all CIFAR-10 training and test data"""
    # Load training data
    all_train_data = []
    all_train_labels = []
    
    for i in range(1, 6):
        batch = unpickle(f"C:/Users/milla/Downloads/cifar-10-python (1)/cifar-10-batches-py/data_batch_{i}")
        data = batch[b'data']
        labels = batch[b'labels']
        
        # Reshape to proper image format (N, C, H, W) and normalize
        data = data.reshape(-1, 3, 32, 32) / 255.0
        all_train_data.append(data)
        all_train_labels.extend(labels)
    
    train_data = np.vstack(all_train_data)
    
    # Load test data
    test_batch = unpickle("C:/Users/milla/Downloads/cifar-10-python (1)/cifar-10-batches-py/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']
    
    # Reshape test data to proper image format and normalize
    test_data = test_data.reshape(-1, 3, 32, 32) / 255.0
    
    return train_data, all_train_labels, test_data, test_labels

def create_parametric_network():
    return nn.Sequential(
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

def create_regular_network():
    return nn.Sequential(
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

class LossTracker:
    def __init__(self):
        self.losses = defaultdict(list)  # {run_id: [losses]}
        self.current_run = 0
        self.lock = threading.Lock()
        
    def add_loss(self, network_type, run_id, loss_history):
        with self.lock:
            key = f"{network_type}_run_{run_id}"
            self.losses[key] = loss_history.copy()
    
    def get_losses(self):
        with self.lock:
            return dict(self.losses)

# Global loss tracker
loss_tracker = LossTracker()

def train_network(network, train_data, train_labels, network_type, run_id, epochs=5, batch_size=512):
    """Train a network and return training time and loss history"""
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    network.train()
    start_time = time.time()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_losses = []
        for batch_idx in range(0, len(train_data), batch_size):
            batch_data = train_data[batch_idx:batch_idx+batch_size]
            batch_labels = train_labels[batch_idx:batch_idx+batch_size]
            
            optimizer.zero_grad()
            outputs = network(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        loss_history.append(avg_epoch_loss)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
    
    # Add to global tracker
    loss_tracker.add_loss(network_type, run_id, loss_history)
    
    end_time = time.time()
    return end_time - start_time, loss_history

def test_network(network, test_data, test_labels, batch_size=256):
    """Test a network and return accuracy"""
    network.eval()
    correct = 0
    
    with torch.no_grad():
        for batch_idx in range(0, len(test_data), batch_size):
            batch_data = test_data[batch_idx:batch_idx+batch_size]
            batch_labels = test_labels[batch_idx:batch_idx+batch_size]
            
            outputs = network(batch_data)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == batch_labels).sum().item()
    
    return correct / len(test_data)

def print_parametric_params(network):
    """Print parameter ranges for parametric network"""
    # Find ParametricGELU layers
    parametric_layers = []
    for i, layer in enumerate(network):
        if isinstance(layer, (ParametricGELU2d, ParametricGELU)):
            parametric_layers.append((i, layer))
    
    # Print first conv and first FC parametric layers
    if len(parametric_layers) >= 2:
        conv_idx, conv_layer = parametric_layers[0]  # First ParametricGELU2d
        fc_idx, fc_layer = None, None
        
        # Find first FC layer
        for idx, layer in parametric_layers:
            if isinstance(layer, ParametricGELU):
                fc_idx, fc_layer = idx, layer
                break
        
        if fc_layer is not None:
            print(f"Conv1 (layer {conv_idx}) - Alpha range: [{conv_layer.alphas.min().item():.4f}, {conv_layer.alphas.max().item():.4f}]")
            print(f"Conv1 (layer {conv_idx}) - Beta range: [{conv_layer.betas.min().item():.4f}, {conv_layer.betas.max().item():.4f}]")
            print(f"FC1 (layer {fc_idx}) - Alpha range: [{fc_layer.alphas.min().item():.4f}, {fc_layer.alphas.max().item():.4f}]")
            print(f"FC1 (layer {fc_idx}) - Beta range: [{fc_layer.betas.min().item():.4f}, {fc_layer.betas.max().item():.4f}]")

def setup_plot():
    """Setup the matplotlib plot"""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.set_title('Parametric (Alpha Only) Network Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Regular Network Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    return fig, ax1, ax2

def update_plot(fig, ax1, ax2):
    """Update the plot with current loss data"""
    losses = loss_tracker.get_losses()
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Setup axes again
    ax1.set_title('Parametric Network (Alpha Only) Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Regular Network Loss Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Plot parametric losses
    param_losses = [(k, v) for k, v in losses.items() if 'Parametric' in k]
    for key, loss_hist in param_losses:
        run_num = key.split('_')[-1]
        epochs = range(1, len(loss_hist) + 1)
        ax1.plot(epochs, loss_hist, alpha=0.7, label=f'Run {run_num}')
    
    # Plot regular losses
    reg_losses = [(k, v) for k, v in losses.items() if 'Regular' in k]
    for key, loss_hist in reg_losses:
        run_num = key.split('_')[-1]
        epochs = range(1, len(loss_hist) + 1)
        ax2.plot(epochs, loss_hist, alpha=0.7, label=f'Run {run_num}')
    
    # Only show legend if we have few runs (to avoid clutter)
    if len(param_losses) <= 10:
        ax1.legend()
    if len(reg_losses) <= 10:
        ax2.legend()
    
    plt.tight_layout()
    plt.pause(0.01)  # Small pause to update the plot

# Main comparison loop
if __name__ == "__main__":
    # Load CIFAR-10 data
    print("Loading CIFAR-10 data...")
    train_data_np, train_labels_list, test_data_np, test_labels_list = load_cifar_data()
    
    # Convert to tensors
    train_data = torch.tensor(train_data_np, dtype=torch.float).to("cuda")
    train_labels = torch.tensor(train_labels_list, dtype=torch.long).to("cuda")
    test_data = torch.tensor(test_data_np, dtype=torch.float).to("cuda")
    test_labels = torch.tensor(test_labels_list, dtype=torch.long).to("cuda")
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print()
    
    # Setup plotting
    fig, ax1, ax2 = setup_plot()
    
    differences = []
    epochs = 5  # Increased epochs for better loss curves
    
    for run in range(500):
        print(f"Run {run+1}/500")
        
        # Create fresh networks for each run
        network_parametric = create_parametric_network()
        network_regular = create_regular_network()
        
        # Train parametric network
        print("Training Parametric network...")
        param_train_time, param_loss_hist = train_network(
            network_parametric, train_data, train_labels, "Parametric", run+1, epochs
        )
        param_accuracy = test_network(network_parametric, test_data, test_labels)
        
        print(f"Parametric - Training time: {param_train_time:.2f}s, Accuracy: {param_accuracy*100:.2f}%")
        print_parametric_params(network_parametric)
        
        # Train regular network
        print("Training Regular network...")
        reg_train_time, reg_loss_hist = train_network(
            network_regular, train_data, train_labels, "Regular", run+1, epochs
        )
        reg_accuracy = test_network(network_regular, test_data, test_labels)
        
        print(f"Regular - Training time: {reg_train_time:.2f}s, Accuracy: {reg_accuracy*100:.2f}%")
        
        difference = param_accuracy - reg_accuracy
        print(f"Difference: {difference*100:.2f}%")
        print("-" * 50)
        
        differences.append(difference)
        
        # Update plot every run
        update_plot(fig, ax1, ax2)
    
    # Final results
    average_difference = sum(differences) / len(differences)
    parametric_wins = sum(1 for d in differences if d > 0)
    
    print(f"\nFinal Results:")
    print(f"Average Difference: {average_difference*100:.2f}%")
    print(f"Parametric wins: {parametric_wins}/500 times")
    print(f"Regular wins: {500-parametric_wins}/500 times")
    print(f"Win rate: {parametric_wins/500*100:.1f}%")
    
    # Keep plot open
    plt.ioff()
    plt.show()
