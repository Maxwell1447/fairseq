import torch
from fairseq import libdual_cuda
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
x = torch.tensor(
    [
        [0, 4, 7, 5, 4] + [8]*2 +[2, 1, 1, 1],
    ]
).to(device)
x = torch.vstack([x]*2)

# m = torch.zeros(2, 10).to(device)
print(x.shape)
t1 = time.time()
# x = x.cpu()
m = libdual_cuda.get_bow_mask_from_sequence(x, 10, 0, 2, 1)
# x = x.cuda()
t2 = time.time()
print(m)
print("time = ", t2-t1)
