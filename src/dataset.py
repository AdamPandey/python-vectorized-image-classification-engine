import torch
from torch.utils.data import Dataset
from typing import Tuple
from src.matrix_ops import CustomMatrixTransforms

class NoiseRobustDataset(Dataset):
    """Custom PyTorch Dataset managing matrix transformation injections."""
    def __init__(self, data: torch.Tensor, labels: torch.Tensor, optimize_pipeline: bool = False):
        self.data = data
        self.labels = labels
        self.optimize_pipeline = optimize_pipeline

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        matrix = self.data[idx]
        label = self.labels[idx]

        if self.optimize_pipeline:
            matrix = CustomMatrixTransforms.manual_normalize(matrix)
        else:
            matrix = CustomMatrixTransforms.adjust_contrast_matrix(matrix)
            matrix = CustomMatrixTransforms.manual_normalize(matrix)

        return matrix, label