from PIL import Image


def get_image_data(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    width, height = image.size

    data = image.getdata()
    data = list(data)

    return data, width, height


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
    list_data, width, height = get_image_data("./Images/Mandelbrot_Fractal_1024x768.png")
    dim = (width, height)
    r, g, b = create_RGB_image(list_data, dim)
    r.save("./Images/red.png")
    g.save("./Images/green.png")
    b.save("./Images/blue.png")

