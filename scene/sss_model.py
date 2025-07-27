""" Subsurface Scattering (SSS) model. """
import math
from typing import Tuple
import torch
from torch import nn

class SSS(nn.Module):
    """ Subsurface Scattering (SSS) model. """
    def __init__(
        self,
        net_width: int = 32,
        factor: int = 2,
        degree: int = 4
    ):
        """ Initializes the Subsurface Scattering (SSS) model.

        Args:
            net_width (int): The width of the neural network layers. Default is 32.
            factor (int): The factor for the network width. Default is 2.
            degree (int): The degree for the positional encoding. Default is 4.
        """
        super().__init__()
        
        # Model parameters
        self.factor = factor
        self.net_width = net_width

        # Positional encoding parameters
        self.degree = degree

        self.main = nn.Sequential(
            nn.Linear(18 + 3 * 2 * self.degree, self.factor * self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width * self.factor, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
        )

        self.residual = nn.Sequential(
            nn.Linear(self.net_width, 3),
            nn.Sigmoid(),
        )

        self.incident_light = nn.Sequential(
            nn.Linear(self.net_width, self.net_width),
            nn.LeakyReLU(),
            nn.Linear(self.net_width, 1),
            nn.ReLU()
        )

    def positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """ Applies positional encoding to the input.

        Args:
            x (torch.Tensor): Input tensor to be positionally encoded.

        Returns:
            torch.Tensor: Positionally encoded tensor.
        """
        result = []
        for d in range(self.degree):
            for fn in [torch.sin, torch.cos]:
                result.append(fn(2.0 ** d * math.pi * x))
        return torch.cat(result, dim=-1)


    def forward(
            self, 
            positions: torch.Tensor,
            rotations: torch.Tensor,
            scales: torch.Tensor,
            view_dirs: torch.Tensor,
            light_dirs: torch.Tensor,
            normals: torch.Tensor,
            visibilities: torch.Tensor,
            light_distances: torch.Tensor, 
            iteration: int = -1
        ) -> Tuple[torch.Tensor, torch.Tensor]: 
        """ Forward pass of the SSS model.

        Args:
            positions (torch.Tensor): Positions of the points. Shape: [batch, 3]
            rotations (torch.Tensor): Rotations of the points. Shape: [batch, 2]
            scales (torch.Tensor): Scales of the points. Shape: [batch, 3]
            view_dirs (torch.Tensor): View directions. Shape: [batch, 3]
            light_dirs (torch.Tensor): Light directions. Shape: [batch, 3]
            normals (torch.Tensor): Normal vectors. Shape: [batch, 3]
            visibilities (torch.Tensor): Visibility values. Shape: [batch, 1]
            light_distances (torch.Tensor): Distances to light sources. Shape: [batch, 1]
            iteration (int): Current iteration number. Default is -1.

        Returns:
            tuple: A tuple containing:
                - residual (torch.Tensor): Residual values. Shape: [batch, 3]
                - incident_light (torch.Tensor): Incident light values. Shape: [batch, 3]
        """

        encoded_positions = self.positional_encoding(positions)

        x = torch.concat([encoded_positions, rotations, scales, view_dirs, light_dirs, normals, visibilities, light_distances], dim=-1)
        x = self.main(x)

        residual = self.residual(x)
        incident_light = self.incident_light(x).repeat(1, 3)
 
        # Linear decay the incident light
        # TODO: This should be moved to args 
        start_iteration = 0
        stop_iteration = 14_000
        max_incident_light = 4.0

        # Calculate midpoint
        midpoint_iteration = (start_iteration + stop_iteration) // 2


        # FIXME: Remove or fix this
        incident_light_addition = 0.0
        if start_iteration <= iteration <= stop_iteration:
            if iteration <= midpoint_iteration:
                # Increasing phase
                incident_light_addition = max_incident_light * (iteration - start_iteration) / (midpoint_iteration - start_iteration)
            else:
                # Decreasing phase
                # incident_light_addition = max_incident_light * (stop_iteration - iteration) / (stop_iteration - midpoint_iteration)
                pass
            
        incident_light = incident_light + incident_light_addition

        return residual, incident_light