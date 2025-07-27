""" Color utils. """
import torch

def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """ Converts sRGB color values to linear color values. 

    Args:
        srgb (torch.Tensor): sRGB color values.

    Returns:
        torch.Tensor: Linear color values.
    """
    return torch.where(srgb > 0.04045, torch.pow((srgb + 0.055) / 1.055, 2.4), srgb / 12.92)

def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """ Converts linear color values to sRGB color values. 

    Args:
        linear (torch.Tensor): Linear color values.

    Returns:
        torch.Tensor: sRGB color values.    
    """
    return torch.where(linear > 0.0031308, 1.055 * torch.pow(linear, 1 / 2.4) - 0.055, 12.92 * linear)