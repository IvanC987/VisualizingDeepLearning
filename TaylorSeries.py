import torch
from torch import nn
from torch.optim import AdamW
import PIL


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")

# Hyperparameters
# -----------------------------------
num_terms = 5
lr = 1e-4

training_iterations = 100
eval_interval = 25
# -----------------------------------

# Using randomized pixel_coords rather than structured as a test later on


def evaluate_model():
    pass


class TSApproximation(nn.Module):
    def __init__(self, image_dim: tuple[int, int, int], num_terms: int = 5):
        super().__init__()

        # Save image dimensions (e.g., (3, 320, 240) for (RGB, width, height)
        self.image_dim = image_dim
        self.num_terms = num_terms

        # Random Initialization for weights of shape (RBG, TaylorSeries Terms, x-y coord) -> (3, num_terms, 2)
        self.coefficients = nn.Parameter(torch.randn(3, num_terms, 2))
        self.bias = nn.Parameter(torch.randn(3))  # Bias term of shape (3) for each color RGB

    def forward(self, x: torch.tensor):
        # First, calculate the number of pixels in an image of each color via width * height
        num_pixels = self.image_dim[1] * self.image_dim[2]

        # Initialize the output tensor of each image's color, fill with corresponding bias. Size = (num_pixels)
        red = torch.full((num_pixels,), self.bias[0].item())
        green = torch.full((num_pixels,), self.bias[1].item())
        blue = torch.full((num_pixels,), self.bias[2].item())

        # Stack it to get (RGB, num_pixels)
        output = torch.stack((red, green, blue))

        for color in range(3):
            for ith_pixel in range(num_pixels):
                # Calculate the Taylor series for each pixel (x[i, 0] for x-coordinate, x[i, 1] for y-coordinate)
                # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
                for jth_term in range(self.num_terms):
                    output[color][ith_pixel] += (
                            self.coefficients[color, jth_term, 0] * (x[ith_pixel, 0] ** jth_term) +  # x-related terms
                            self.coefficients[color, jth_term, 1] * (x[ith_pixel, 1] ** jth_term)  # y-related terms
                    )

        # Reshape to original image dimensions
        return output.view(self.image_dim)


image_dimensions = (3, 320, 240)  # RGB, Width, Height
pixel_coords = torch.tensor([[(x, y) for y in range(image_dimensions[2])] for x in range(image_dimensions[1])], dtype=torch.float32).reshape(-1, 2)

model = TSApproximation(image_dimensions, num_terms=num_terms)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=lr)


for step in range(training_iterations):
    y_pred = model(pixel_coords)
    loss = None

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == training_iterations - 1:
        evaluate_model()




