import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np
import time

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class ParametricGELU2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.alphas = nn.Parameter(torch.zeros(out_channels))
        self.betas = nn.Parameter(torch.ones(out_channels))
        
    def forward(self, x):
        x = self.conv(x)
        # Reshape for broadcasting: (batch, channels, height, width)
        alphas = self.alphas.view(1, -1, 1, 1)
        betas = self.betas.view(1, -1, 1, 1)
        gelu_input = (x-alphas)
        return F.gelu(gelu_input)

class ParametricGELU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.alphas = nn.Parameter(torch.zeros(output_dim))
        self.betas = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x):
        linear_out = self.linear(x)
        gelu_input = self.betas * (linear_out - self.alphas)
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
        nn.BatchNorm2d(64),
        ParametricGELU2d(64, 64),  # 32x32x64
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2, 2),  # 16x16x64

        # Second convolutional block
        ParametricGELU2d(64, 128),  # 16x16x128
        nn.BatchNorm2d(128),
        ParametricGELU2d(128, 128),  # 16x16x128
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2, 2),  # 8x8x128

        # Third convolutional block
        ParametricGELU2d(128, 256),  # 8x8x256
        nn.BatchNorm2d(256),
        ParametricGELU2d(256, 256),  # 8x8x256
        nn.BatchNorm2d(256),
        nn.MaxPool2d(2, 2),  # 4x4x256

        # Flatten and fully connected layers
        nn.Flatten(),  # 44256 = 4096
        ParametricGELU(4096, 1024),
        nn.BatchNorm1d(1024),

        ParametricGELU(1024, 1024),
        nn.BatchNorm1d(1024),

        nn.Linear(1024, 10)
    ).to("cuda")

def create_regular_network():
    return nn.Sequential(
        # First convolutional block
        nn.Conv2d(3, 64, kernel_size=3, padding=1),  # 32x32x64
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 32x32x64
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(2, 2),  # 16x16x64

        # Second convolutional block
        nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 16x16x128
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),  # 16x16x128
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.MaxPool2d(2, 2),  # 8x8x128

        # Third convolutional block
        nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 8x8x256
        nn.BatchNorm2d(256),
        nn.GELU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 8x8x256
        nn.BatchNorm2d(256),
        nn.GELU(),
        nn.MaxPool2d(2, 2),  # 4x4x256

        # Flatten and fully connected layers
        nn.Flatten(),  # 44256 = 4096
        nn.Linear(4096, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),

        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.GELU(),

        nn.Linear(1024, 10)
    ).to("cuda")

def train_network(network, train_data, train_labels, epochs=5, batch_size=512):
    """Train a network and return training time"""
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    network.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        for batch_idx in range(0, len(train_data), batch_size):
            batch_data = train_data[batch_idx:batch_idx+batch_size]
            batch_labels = train_labels[batch_idx:batch_idx+batch_size]
            
            optimizer.zero_grad()
            outputs = network(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    end_time = time.time()
    return end_time - start_time

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

def create_networks_with_same_init():
    """Create both networks with identical initialization for fair comparison"""
    # Create regular network first
    network_regular = create_regular_network()
    network_parametric = create_parametric_network()
    
    # Copy weights from regular to parametric (matching layers only)
    regular_layers = list(network_regular.children())
    parametric_layers = list(network_parametric.children())
    
    regular_idx = 0
    for param_layer in parametric_layers:
        if isinstance(param_layer, ParametricGELU2d):
            # Find corresponding Conv2d in regular network
            while regular_idx < len(regular_layers) and not isinstance(regular_layers[regular_idx], nn.Conv2d):
                regular_idx += 1
            if regular_idx < len(regular_layers):
                # Copy conv weights and biases
                param_layer.conv.weight.data.copy_(regular_layers[regular_idx].weight.data)
                if param_layer.conv.bias is not None and regular_layers[regular_idx].bias is not None:
                    param_layer.conv.bias.data.copy_(regular_layers[regular_idx].bias.data)
                regular_idx += 1
                
        elif isinstance(param_layer, ParametricGELU):
            # Find corresponding Linear in regular network
            while regular_idx < len(regular_layers) and not isinstance(regular_layers[regular_idx], nn.Linear):
                regular_idx += 1
            if regular_idx < len(regular_layers):
                # Copy linear weights and biases
                param_layer.linear.weight.data.copy_(regular_layers[regular_idx].weight.data)
                if param_layer.linear.bias is not None and regular_layers[regular_idx].bias is not None:
                    param_layer.linear.bias.data.copy_(regular_layers[regular_idx].bias.data)
                regular_idx += 1
                
        elif isinstance(param_layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear)):
            # Copy other layers directly
            while regular_idx < len(regular_layers) and type(regular_layers[regular_idx]) != type(param_layer):
                regular_idx += 1
            if regular_idx < len(regular_layers):
                param_layer.load_state_dict(regular_layers[regular_idx].state_dict())
                regular_idx += 1
    
    return network_parametric, network_regular

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
        fc_idx, fc_layer = parametric_layers[-2]     # First ParametricGELU FC layer
        
        print(f"Conv1 (layer {conv_idx}) - Alpha range: [{conv_layer.alphas.min().item():.4f}, {conv_layer.alphas.max().item():.4f}]")
        print(f"Conv1 (layer {conv_idx}) - Beta range: [{conv_layer.betas.min().item():.4f}, {conv_layer.betas.max().item():.4f}]")
        print(f"FC1 (layer {fc_idx}) - Alpha range: [{fc_layer.alphas.min().item():.4f}, {fc_layer.alphas.max().item():.4f}]")
        print(f"FC1 (layer {fc_idx}) - Beta range: [{fc_layer.betas.min().item():.4f}, {fc_layer.betas.max().item():.4f}]")
    """Print parameter ranges for parametric network"""
    # Find ParametricGELU layers
    parametric_layers = []
    for i, layer in enumerate(network):
        if isinstance(layer, (ParametricGELU2d, ParametricGELU)):
            parametric_layers.append((i, layer))
    
    # Print first conv and first FC parametric layers
    if len(parametric_layers) >= 2:
        conv_idx, conv_layer = parametric_layers[0]  # First ParametricGELU2d
        fc_idx, fc_layer = parametric_layers[-2]     # First ParametricGELU FC layer
        
        print(f"Conv1 (layer {conv_idx}) - Alpha range: [{conv_layer.alphas.min().item():.4f}, {conv_layer.alphas.max().item():.4f}]")
        print(f"Conv1 (layer {conv_idx}) - Beta range: [{conv_layer.betas.min().item():.4f}, {conv_layer.betas.max().item():.4f}]")
        print(f"FC1 (layer {fc_idx}) - Alpha range: [{fc_layer.alphas.min().item():.4f}, {fc_layer.alphas.max().item():.4f}]")
        print(f"FC1 (layer {fc_idx}) - Beta range: [{fc_layer.betas.min().item():.4f}, {fc_layer.betas.max().item():.4f}]")

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
    
    differences = []
    
    for run in range(500):
        print(f"Run {run+1}/500")
        
        # Create fresh networks with same initialization for each run
        network_parametric, network_regular = create_networks_with_same_init()
        
        # Train parametric network
        print("Training Parametric network...")
        param_train_time = train_network(network_parametric, train_data, train_labels)
        param_accuracy = test_network(network_parametric, test_data, test_labels)
        
        print(f"Parametric - Training time: {param_train_time:.2f}s, Accuracy: {param_accuracy*100:.2f}%")
        print_parametric_params(network_parametric)
        
        # Train regular network
        print("Training Regular network...")
        reg_train_time = train_network(network_regular, train_data, train_labels)
        reg_accuracy = test_network(network_regular, test_data, test_labels)
        
        print(f"Regular - Training time: {reg_train_time:.2f}s, Accuracy: {reg_accuracy*100:.2f}%")
        
        difference = param_accuracy - reg_accuracy
        print(f"Difference: {difference*100:.2f}%")
        print("-" * 50)
        
        differences.append(difference)
    
    # Final results
    average_difference = sum(differences) / len(differences)
    parametric_wins = sum(1 for d in differences if d > 0)
    
    print(f"\nFinal Results:")
    print(f"Average Difference: {average_difference*100:.2f}%")
    print(f"Parametric wins: {parametric_wins}/25 times")
    print(f"Regular wins: {25-parametric_wins}/25 times")
    print(f"\n All differences: {differences}")
