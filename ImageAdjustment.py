from PIL import Image


def print_resolution(image_path):
    image = Image.open(image_path)

    width, height = image.size
    print(f"Resolution: {width}x{height}")


def change_resolution(image_path, new_resolution, save_path="temp.png"):
    image = Image.open(image_path)
    low_res_image = image.resize(new_resolution)

    # Save the resized image to a new file
    low_res_image.save(save_path)

    print(f"Low-resolution image saved as: {save_path}")


if __name__ == "__main__":
    img_path = f"./Mandelbrot_Fractal_3840x2160.png"
    print_resolution(img_path)
    change_resolution(img_path, (32, 24))

