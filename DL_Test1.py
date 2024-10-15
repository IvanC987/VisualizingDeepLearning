import torch

dimensions = [9, 6]
width, height = dimensions[1], dimensions[0]
patch_size = 3

a = [i for i in range(dimensions[0] * width)]

original = torch.tensor(a)

a = torch.tensor(a)

temp = a.reshape(dimensions)

print(temp)

print("\n\n\n\n")

for i in range(len(a)):
    coords = [i//width, i % width]
    a[i] = ((coords[0]//patch_size) * (width//patch_size)) + (coords[1] // patch_size)
    print(f"{i=}, {a[i]}, {coords=}")


# for i in range(len(a)):
#     a[i] = (i//patch_size**2) + ((i % width) // patch_size)
#     print(f"{i=}, {a[i]}")


print(original.reshape(dimensions))
print(a.reshape(dimensions))
