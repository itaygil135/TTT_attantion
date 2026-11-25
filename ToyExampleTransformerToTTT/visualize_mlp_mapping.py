"""In this module, we visualize a toy MLP mapping from 2D input space to 3D output space (grid to color image)."""

import numpy as np
import matplotlib.pyplot as plt

def create_image(mlp, min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0, resolution=256):
    """Create an image by passing coordinates through the MLP."""
    x = np.linspace(min_x, max_x, resolution)
    y = np.linspace(min_y, max_y, resolution)
    X, Y = np.meshgrid(x, y)
    coordinates = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        output = mlp(coordinates)
    output = output.numpy().reshape((resolution, resolution, 3))
    output = (output - output.min()) / (output.max() - output.min())
    return output

import torch
import torch.nn as nn
from torch.nn import functional as F

class Simple1LayerMLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x, dim=-1)
        x = self.fc2(x)
        return x

    def init_with_KV(self, K, V):
        with torch.no_grad():
            self.fc1.weight.copy_(K)
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(V)
            self.fc2.bias.zero_()

def visualize_attention_mlp(K, V):
    """Visualize the MLP mapping of softmax(xK)V."""

    mlp = Simple1LayerMLP(input_dim=2, hidden_dim=len(K), output_dim=3, activation=F.softmax)
    mlp.init_with_KV(K, V)

    image = create_image(mlp, resolution=1024)
    
    # save the image
    plt.imsave("mlp_mapping.png", image)

    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    # Example K and V matrices
    K = torch.rand(8, 2) * 2 - 1  # 2 hidden units, 2D input
    # V should be defined manually with different colors
    V = torch.tensor([[1.0, 0.0, 0.0],   # Red
                      [0.0, 1.0, 0.0],   # Green
                      [0.0, 0.0, 1.0],   # Blue
                      [1.0, 1.0, 0.0],   # Yellow
                      [1.0, 0.0, 1.0],   # Magenta
                      [0.0, 1.0, 1.0],   # Cyan
                      [1.0, 1.0, 1.0],   # White
                      [0.0, 0.0, 0.0]]).T  # Black

    visualize_attention_mlp(K, V)