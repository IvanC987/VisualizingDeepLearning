import torch
from torch import nn


# Using randomized pixel_coords rather than structured as a test later on

class TSApproximation(nn.Module):
    def __init__(self, image_dim: tuple[int, int], num_terms: int = 5):
        super().__init__()

        # Save image dimensions (e.g., (320, 240))
        self.image_dim = image_dim
        self.num_terms = num_terms

        self.coefficients = nn.Parameter(torch.randn(num_terms, 2))  # Random Initialization for weights
        self.bias = nn.Parameter(torch.randn(1))  # Bias term

    def forward(self, x: torch.tensor):
        output = torch.zeros(x.size(0)) + self.bias.item()  # Initializes output to zeros and adds bias

        for i in range(x.size(0)):
            # Calculate the Taylor series for each pixel (x[i, 0] for x-coordinate, x[i, 1] for y-coordinate)
            for j in range(self.num_terms):
                output[i] += (
                        self.coefficients[j, 0] * (x[i, 0] ** j) +  # x-related terms
                        self.coefficients[j, 1] * (x[i, 1] ** j)  # y-related terms
                )

        return output.view(self.image_dim)  # Reshape to original image dimensions


image_dimensions = (10, 5)
model = TSApproximation(image_dimensions)
pixel_coords = torch.tensor([[(x, y) for y in range(image_dimensions[1])] for x in range(image_dimensions[0])], dtype=torch.float32).reshape(-1, 2)


