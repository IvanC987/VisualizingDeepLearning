import torch



a = [i for i in range(21)]

a = torch.tensor(a)
a = a.reshape(3, -1)

print(a)
print(a.shape)

print("----------------")

b = a.permute(-1, -2)
print(b)
print(b.shape)


