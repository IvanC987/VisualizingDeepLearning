import torch
from torch import nn
from torch.optim import AdamW
from ImageTransform import get_image_data, create_image, create_RGB_image
import time


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")

# Hyperparameters
# -----------------------------------
num_terms = 5
lr = 1e-2

training_iterations = 100
eval_interval = 5

image_path = "./Images/Mandelbrot_Fractal_320x240.png"
save_image_path = "./CreatedImages/"
# -----------------------------------

# Using randomized pixel_coords rather than structured as a test later on


def init_data():
    image_data, width, height = get_image_data(image_path)
    image_data = torch.tensor(image_data, dtype=torch.float32)

    image_dimensions = (3, width, height)

    # Pixel Coordinates would be a tensor of shape (num_pixels, 2)
    pixel_coords = torch.tensor([[(x, y) for y in range(image_dimensions[2])] for x in range(image_dimensions[1])], dtype=torch.float32).reshape(-1, 2)

    # Now normalize, since they can be quite large and throw off the model
    pixel_coords[:, 0] /= width
    pixel_coords[:, 1] /= height


    # Since image_data is of shape width * height, where each element is (R, G, B) value of each pixel, this would need to be reshaped into the desired format
    # Which would be splitting the image into the shape (color, width, height)
    red_data = image_data[:, 0]
    green_data = image_data[:, 1]
    blue_data = image_data[:, 2]

    target_data = torch.stack((red_data, green_data, blue_data))  # Normalizing the values to be between 0 and 1
    return image_dimensions, pixel_coords, target_data


def save_image(img_path):
    pass


class TSApproximation(nn.Module):
    def __init__(self, image_dim: tuple[int, int, int], num_terms: int = 5):
        super().__init__()

        # Save image dimensions (e.g., (3, 320, 240) for (RGB, width, height)
        self.image_dim = image_dim
        self.num_terms = num_terms

        # Random Initialization for weights of shape (RBG, TaylorSeries Terms, x-y coord) -> (3, num_terms, 2)
        self.coefficients = nn.Parameter(torch.randn(3, num_terms, 2))
        nn.init.xavier_uniform_(self.coefficients)

        # Bias term of shape (3) for each color RGB
        self.bias = nn.Parameter(torch.randn(3))
        nn.init.zeros_(self.bias)

    def forward(self, pix_coord: torch.tensor):
        # First, calculate the number of pixels in an image of each color via width * height
        num_pixels = self.image_dim[1] * self.image_dim[2]

        # Initialize the output tensor of each image's color, fill with corresponding bias. Size = (num_pixels)
        red_bias = torch.full((num_pixels,), self.bias[0].item())
        green_bias = torch.full((num_pixels,), self.bias[1].item())
        blue_bias = torch.full((num_pixels,), self.bias[2].item())

        # Stack it to get (RGB, num_pixels)
        output = torch.stack((red_bias, green_bias, blue_bias))

        # for color in range(3):
        #     for ith_pixel in range(num_pixels):
        #         # Calculate the Taylor series for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
        #         # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
        #         for jth_term in range(self.num_terms):
        #             output[color][ith_pixel] += (
        #                     self.coefficients[color, jth_term, 0] * (pix_coord[ith_pixel, 0] ** jth_term) +  # x-related terms
        #                     self.coefficients[color, jth_term, 1] * (pix_coord[ith_pixel, 1] ** jth_term)  # y-related terms
        #             )

        powers = torch.arange(self.num_terms, dtype=torch.float32, device=device)
        for color in range(3):
            for ith_pixel in range(num_pixels):
                # Calculate the Taylor series for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
                # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
                output[color][ith_pixel] += (
                    torch.sum(self.coefficients[color, :, 0] * (pix_coord[ith_pixel, 0] ** powers)) +
                    torch.sum(self.coefficients[color, :, 1] * (pix_coord[ith_pixel, 1] ** powers))
                )


        # Return the output, shape=(3, num_pixels)
        return output



image_dimensions, pixel_coords, target_data = init_data()
pixel_coords = pixel_coords.to(device)
target_data = target_data.to(device)


model = TSApproximation(image_dimensions, num_terms=num_terms).to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=lr)
print(f"There are {sum(p.numel() for p in model.parameters())} parameters in this model")


start = time.time()
prev_loss = 0
print("Starting")
image_index = 0
for step in range(training_iterations):
    y_pred = model(pixel_coords)
    loss = criterion(y_pred, target_data)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0 or step == training_iterations - 1:
        save_image(save_image_path + f"image{image_index}.png")
        print(f"Currently at step={step}   |   loss={int(loss.item())//1}O   |   time={time.time() - start:.1f}s   |   Loss_decrease={"N/A-" if prev_loss == 0 else round((1 - (loss.item()/prev_loss)) * 100, 2)}%")
        prev_loss = loss.item()
        start = time.time()
        image_index += 1


# Remember to scale output by 255 when creating resulting image!

"""
235.0T, 186.0T, 304.0T
317.0T, 196.0T, 



For *1
loss=163.0T

For /3
loss=8.0T

For /10
loss=1652.0B

For /30
loss=171.0B

For /100
loss=7.0B

For /300
loss=2.0B

For /3000
loss=49.0M

For /10_000
loss=630.0K

For /25_000
loss=207561
"""

