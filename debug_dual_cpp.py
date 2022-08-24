import torch
from fairseq import libdual_cuda

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor(
    [
        [0, 4, 7, 5, 4] + [8]*200 +[2, 1, 1, 1],
    ]
).to(device)
x = torch.vstack([x]*10)

# m = torch.zeros(2, 10).to(device)
print(x.shape)
m = libdual_cuda.get_bow_mask_from_sequence(x, 10, 0, 2, 1)

print(m)
