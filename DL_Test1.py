import torch


t = (2, 3)

pixel_coords = [[(x, y) for y in range(t[1])] for x in range(t[0])]

for p in pixel_coords:
    print(p)
