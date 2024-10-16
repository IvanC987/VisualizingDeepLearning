import torch
from torch import nn
from torch.optim import AdamW
from ImageTransform import get_image_data, create_image, create_RGB_image
import time
import random


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Currently using {device=}")

# Hyperparameters
# -----------------------------------
num_terms = 10
lr = 0.05

training_iterations = 1500
eval_interval = 1

patch_size = 5
image_path = "./Images/Mandelbrot_Fractal_1024x768.png"
save_image_path = "./CreatedImages/"
# -----------------------------------

# Using randomized pixel_coords rather than structured as a test later on


def init_data():
    image_data, image_dimensions = get_image_data(image_path, patch_size)

    # Pixel Coordinates would be a tensor of shape (num_pixels, 2)
    pixel_coords = torch.tensor([[(x, y) for y in range(image_dimensions[1])] for x in range(image_dimensions[0])], dtype=torch.float32).reshape(-1, 2)

    # Normalize the coordinate values in the range of (0-1)
    pixel_coords[:, 0] /= image_dimensions[0]
    pixel_coords[:, 1] /= image_dimensions[1]


    # Since image_data is of shape width * height, where each element is (R, G, B) value of each pixel, this would need to be reshaped into the desired format
    # Which would be splitting the image into the shape (color, width, height)
    red_data = image_data[:, 0]
    green_data = image_data[:, 1]
    blue_data = image_data[:, 2]

    # Stacking the tensors and normalizing the values to be between 0 and 1
    target_data = torch.stack((red_data, green_data, blue_data))
    return image_dimensions, pixel_coords, target_data


def save_image(data: torch.tensor, image_dim, img_path: str):
    # Remember, data is the output of the forward pass, of shape (color, num_pixels)
    # Goal is to reshape it into typical RGB format of shape (num_pixels, color)
    data = data.permute(-1, -2).tolist()
    data = [tuple([round(i) for i in e]) for e in data]

    image = create_image(data, image_dim)
    image.save(img_path)



class TSApproximation(nn.Module):
    def __init__(self, image_dim: tuple[int, int], num_terms: int = 5):
        super().__init__()

        self.image_dim = image_dim
        self.num_terms = num_terms

        # Calculate the number of patches
        num_patches = (image_dim[0] // patch_size) * (image_dim[1] // patch_size)
        # Random Initialization for weights of shape (RBG, TaylorSeries Terms, x-y coord) -> (3, num_terms, 2)
        self.coefficients = nn.Parameter(torch.randn(num_patches, 3, num_terms, 2))
        nn.init.xavier_uniform_(self.coefficients)

        # Bias term of shape (3) for each color RGB
        self.bias = nn.Parameter(torch.randn(num_patches, 3))
        nn.init.zeros_(self.bias)

    def forward(self, pix_coord: torch.tensor):
        # First, calculate the number of pixels in an image of each color via width * height
        num_pixels = self.image_dim[0] * self.image_dim[1]

        # # Initialize the output tensor of each image's color, fill with corresponding bias. Size = (num_pixels)
        # red_bias = torch.full((num_pixels,), self.bias[0].item())
        # green_bias = torch.full((num_pixels,), self.bias[1].item())
        # blue_bias = torch.full((num_pixels,), self.bias[2].item())
        # # Stack it to get (RGB, num_pixels)
        # output = torch.stack((red_bias, green_bias, blue_bias))

        output = torch.zeros(3, num_pixels, device=device)

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
                # Calculate the patch index. There's a video that explains the logic for calculating the patch index.
                width = self.image_dim[0]
                coordinate = [ith_pixel//width, ith_pixel % width]
                patch_index = ((coordinate[0]//patch_size) * (width//patch_size)) + (coordinate[1] // patch_size)

                # Calculate the Taylor series for each pixel (pix_coord[i, 0] for x-coordinate, pix_coord[i, 1] for y-coordinate)
                # This loop can be vectorized to improve efficiency, but for sake of simplicity, will leave it be until optimization bottleneck occurs
                output[color][ith_pixel] = (
                    torch.sum(self.coefficients[patch_index, color, :, 0] * (pix_coord[ith_pixel, 0] ** powers)) +  # Corresponds to X values
                    torch.sum(self.coefficients[patch_index, color, :, 1] * (pix_coord[ith_pixel, 1] ** powers)) +  # Corresponds to Y values
                    self.bias[patch_index, color]  # The associated bias with this current patch
                )


        # Return the output, shape=(3, num_pixels)
        return output



image_dimensions, pixel_coords, target_data = init_data()
pixel_coords = pixel_coords.to(device)
target_data = target_data.to(device)

# Save the adjusted target image
save_image(target_data, image_dimensions, f"{save_image_path}Target.png")


model = TSApproximation(image_dimensions, num_terms=num_terms).to(device)
model = model.half()
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=lr)

num_params = sum(p.numel() for p in model.parameters())
if num_params < 1000:
    print(f"There are {num_params} parameters in this model")
elif 1000 <= num_params <= 1e6:
    print(f"There are {num_params/1000:.2f}K parameters in this model")
else:
    print(f"There are {num_params/1e6:.2f}M parameters in this model")


start = time.time()
prev_loss = 0
print("Starting")
image_index = 0
for step in range(training_iterations):
    with torch.amp.autocast(device):
        y_pred = model(pixel_coords)
    loss = criterion(y_pred, target_data)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


    if step % eval_interval == 0 or step == training_iterations - 1:
        save_image(y_pred, image_dimensions, f"{save_image_path}image{image_index}.png")
        print(f"Currently at step={step}   |   loss={loss.item():.2f}   |   time={time.time() - start:.1f}s   |   Loss_decrease={'N/A-' if prev_loss == 0 else round((1 - (loss.item()/prev_loss)) * 100, 2)}%")
        prev_loss = loss.item()
        start = time.time()
        image_index += 1
        print(y_pred.shape)
        print(y_pred[0, 2400:2420])
        print(target_data[0, 2400:2420])
        print("\n\n\n")



# Remember to scale output by 255 when creating resulting image!
