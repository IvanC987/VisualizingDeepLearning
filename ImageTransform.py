from PIL import Image
import torch


def get_image_data(image_path, patch_size):
    image = Image.open(image_path)
    image = image.convert("RGB")
    image_dim = image.size


    data = list(image.getdata())  # Original shape (pixel, color)
    pad_x = (patch_size - (image_dim[0] % patch_size)) % patch_size
    pad_y = (patch_size - (image_dim[1] % patch_size)) % patch_size


    data = torch.tensor(data, dtype=torch.float32).reshape(image_dim[1], image_dim[0], 3)

    # Permute the tensor then pad accordingly. Using white at the moment for a more visible contrast on my end.
    data = torch.nn.functional.pad(data.permute(2, 0, 1), (0, pad_x, 0, pad_y), mode='constant', value=255)
    # Permute the padded tensor back into its original format
    data = data.permute(1, 2, 0)

    data = data.reshape(-1, 3)  # Return it back to original format of (pixel, color) shape
    image_dim = (image_dim[0] + pad_x, image_dim[1] + pad_y)


    # Return shape tensor(pixel, color)
    return data, image_dim



def create_image(data, image_dim):
    image = Image.new("RGB", image_dim)
    image.putdata(data)
    return image


def create_RGB_image(data, image_dim):
    red = Image.new("RGB", image_dim)
    green = Image.new("RGB", image_dim)
    blue = Image.new("RGB", image_dim)

    red_data = [(r[0], 0, 0) for r in data]
    green_data = [(0, g[1], 0) for g in data]
    blue_data = [(0, 0, b[2]) for b in data]

    red.putdata(red_data)
    green.putdata(green_data)
    blue.putdata(blue_data)

    return red, green, blue


if __name__ == "__main__":
    # list_data, image_dimension = get_image_data("./Images/Mandelbrot_Fractal_320x240.png", 10)
    # r, g, b = create_RGB_image(list_data, image_dimension)
    # r.save("./Images/red.png")
    # g.save("./Images/green.png")
    # b.save("./Images/blue.png")

    data, image_dim = get_image_data("./Images/Mandelbrot_Fractal_320x240.png", 100)
    data = [tuple([int(e) for e in d]) for d in data.tolist()]

    image = create_image(data, image_dim)
    image.save("T1.png")

