from fairseq import utils
import torch


if __name__ == "__main__":
    x = torch.tensor([
        0,1,2,2,4,5,
        0,1,2,4,5,6,6])
    orig = torch.tensor([
        0,0,0,0,0,0,
        1,1,1,1,1,1,1])
    t = torch.tensor([0,1,1,2,6])

    res = utils.get_precision_score(x, t, orig)

    print(res)