import torch

class CustomMatrixTransforms:
    """
    Implements mathematical matrix transformations directly on raw PyTorch tensors 
    to preprocess and clean noisy visual data matrices.
    """
    @staticmethod
    def manual_normalize(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
        """Applies mathematical matrix standardization: (X - mu) / sigma."""
        return (tensor - mean) / std

    @staticmethod
    def adjust_contrast_matrix(tensor: torch.Tensor, factor: float = 1.2) -> torch.Tensor:
        """
        Performs scalar matrix multiplication and clipping constraints to enhance 
        the contrast of noisy pixel matrices.
        """
        mean = torch.mean(tensor)
        # Structural matrix linear blending formula
        enhanced_tensor = (tensor - mean) * factor + mean
        return torch.clamp(enhanced_tensor, 0.0, 1.0)