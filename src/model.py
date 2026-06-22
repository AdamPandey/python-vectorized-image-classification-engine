import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxClassifierNet(nn.Module):
    """A clean, robust PyTorch Neural Network architecture for classification sweeps."""
    def __init__(self):
        super(SoftmaxClassifierNet, self).__init__()
        # Linear layer inputs mapping a flattened 8x8 matrix (64 features) to hidden dimensions
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2) # Binary outcome classification layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flattening matrix tensor structure natively
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x