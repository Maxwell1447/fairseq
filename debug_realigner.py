import sys
import time
import torch
import timeit

from fairseq import realigner as realigner_module
from fairseq.models.nat.multi_levenshtein_utils import apply_plh


def test_1(Kmax=64, M=65, use_alpha_max=False):

    torch.random.manual_seed(0)

    B = 3
    N = 3
    L = 5
    p_cost = 0.2
    r_cost = 1.
    alpha = 0.5

    logits = -(torch.arange(0, Kmax + 1, dtype=torch.float32) ** 1.5) * 0.2 - 0.1
    # logits = -torch.tensor([0.2, 0.6, 1.2])

    logits = logits[None, None, None, :].expand(B, N, L-1, Kmax + 1).float().clone()
    # print(logits)

    # sys.exit(0)
    # logits = -torch.rand(B, N, L - 1, Kmax + 1)
    # x = torch.ones(B, N, L)
    # x = torch.randint(4, 8, (B, N, L))
    # x[:, :, -1] = 2
    # x[:, :, 0] = 0
    x = torch.tensor([
        [
            [0, 7, 5, 4, 2],
            [0, 7, 7, 5, 2],
            [0, 6, 2, 1, 1]
        ],
        [
            [0, 7, 5, 4, 2],
            [0, 7, 7, 5, 2],
            [0, 6, 6, 6, 2]
        ],
        [
            [0, 7, 5, 4, 2],
            [0, 7, 7, 5, 2],
            [0, 6, 2, 1, 1]
        ],
    ])

    t1 = time.time()
    realigner = realigner_module.RealignBatch(
        x,
        logits,
        p_cost,
        r_cost,
        alpha,
        2,
        M,
        use_alpha_max
    )
    Dt = time.time() - t1

    # print("success")
    # success_mask = realigner.get_success_mask().bool()
    # print(success_mask)

    # plh_pred = realigner.get_realigned_plh_pred()

    # print("x")
    # print(x.numpy())
    # print("logits")
    # print(logits.numpy())
    # print("plh pred")
    # print(plh_pred.numpy())

    # x_mod, _ = apply_plh(x, None, plh_pred, 1, 3, 2)

    # print("modified")
    # print(x_mod.numpy())
    print("-", end="", flush=True)

    return Dt

def test_2(B=3, N=3, L=5, Kmax=64, M=65, use_alpha_max=False):

    torch.random.manual_seed(0)
    V = 100

    p_cost = 0.2
    r_cost = 1.
    alpha = 0.5

    y_tgt = torch.randint(4, V + 4, (B, L))
    y_tgt[:, 0] = 0
    y_tgt[:, -1] = 2
    # print(y_tgt)

    

    lens = torch.randint(2, min(2 + int(L * 0.6), L + 1), (B * N,))

    mask = torch.arange(L)[None, :].expand(B * N, L) < lens[:, None]
    # mask.view(B, N, L)
    
    idx = torch.arange(L - 2)[None, :].expand(B * N, L - 2)
    shuffler_idx = torch.rand(B * N, L - 2).argsort(-1)
    batch_idx = torch.arange(B * N)[:, None].expand(B * N, L - 2)
    mask_shuffled = torch.ones(B * N, L, dtype=bool)
    mask_shuffled[:, 1:-1] = idx[batch_idx, shuffler_idx] < (lens[:, None] - 2)
    # mask_shuffled[:, 0] = True
    # mask_shuffled[:, -1] = True

    # print("mask_shuffled\n", mask_shuffled)

    # lens_ins = (torch.rand(B * N) * (L - lens + 0.5) + lens).long()
    lens_ins = lens
    mask_ins = torch.arange(L)[None, :].expand(B * N, L) < lens_ins[:, None]

    # print("lens", lens)
    # print("lens_ins", lens_ins)

    y_noise = torch.ones(B * N, L, dtype=torch.long)

    # mask_shuffled_left = ~torch.sort(~mask_shuffled, -1)[0]

    # print("mask shuffled left", mask_shuffled_left)

    idx = torch.arange(1, L - 1)[None, :].expand(B * N, L - 2)
    shuffler = torch.rand(B * N, L - 2)
    shuffler[~mask_ins[:, 2:]] = 1.
    # shuffler[:, 0] = 1.
    # print(torch.arange(B * N)[:, None])
    # print((lens_ins - 1)[None, :])
    # shuffler[torch.arange(B * N), (lens_ins - 1)] = 1.
    # print("shuffler")
    # print(shuffler)
    shuffler_idx = shuffler.sort(dim=-1, descending=False, stable=True)[1]
    # print("shufler idx\n", shuffler_idx)
    batch_idx = torch.arange(B * N)[:, None].expand(B * N, L - 2)

    # print("sorted\n", idx[batch_idx, shuffler_idx])

    mask_shuffled_ins = torch.zeros(B * N, L, dtype=bool)

    mask_shuffled_ins[:, 1:-1] = idx[batch_idx, shuffler_idx] < (lens[:, None] - 1)

    # print("mask_shuffled_ins\n", mask_shuffled_ins)
    mask_shuffled_ins[:, 0] = True
    mask_shuffled_ins[torch.arange(B * N), (lens_ins - 1)] = True
    
    # mask_shuffled_ins = idx[batch_idx, shuffler_idx] < (lens[:, None])

    # print("mask_shuffled_ins\n", mask_shuffled_ins)

    y_tgt_expanded = y_tgt[:, None, :].expand(B, N, L).reshape(B * N, L)
    # print("y init\n", y_tgt_expanded.view(B, N, L))


    y_noise[mask_shuffled_ins] = y_tgt_expanded[mask_shuffled]

    mask_ins_rd = mask_ins & ~mask_shuffled_ins

    # print(y_noise)

    y_noise[mask_ins_rd] = torch.randint(4, V + 4, (mask_ins_rd.sum().item(), ))
    y_noise[~mask_ins] = 1

    x = y_noise.view(B, N, L).clone()

    # print("mask ins\n", mask_ins.long())

    # print(y_tgt)
    # print(x)

    # print(mask_shuffled)
    cum = (~mask_shuffled).cumsum(-1)
    # print(cum)
    delta = cum[mask_shuffled][1:] - cum[mask_shuffled][:-1]
    delta = delta[delta >= 0]

    # print(delta)

    logit_idx = torch.zeros(B, N, L - 1, dtype=torch.long)
    mask_logits = x[:, :, :-1].ne(1) & x[:, :, :-1].ne(2)
    logit_idx[mask_logits] = delta

    # print(logit_idx)

    noise_strength = 0.6
    var_strength = 1.0
    min_var = 0.2

    num_logits = mask_logits.sum()
    mu = (delta.float() + torch.randn_like(delta.float()) * noise_strength)

    # print(delta)
    # print(mu)

    var = min_var + torch.rand_like(delta.float()) * var_strength

    # print(var)

    ks = torch.arange(Kmax + 1, dtype=torch.float)

    # print(ks[None, :] - mu[:, None 

    logits = torch.zeros(B, N, L - 1, Kmax + 1)

    logits[mask_logits] = -(ks[None, :] - mu[:, None]) ** 2 / var[:, None]
    logits[mask_logits] -= torch.logsumexp(logits[mask_logits], -1)[:, None]

    # print(logits)

    # print(logits[mask_logits])

    # print(torch.exp(logits[mask_logits]))

    # print(x.shape, logits.shape)

    #############################
    t1 = time.time()
    realigner = realigner_module.RealignBatch(
        x,
        logits,
        p_cost,
        r_cost,
        alpha,
        2,
        M,
        use_alpha_max
    )
    Dt = time.time() - t1

    print("*", end="", flush=True)

    return Dt

def test_3(B=3, N=3, L=5, Kmax=64, M=65, use_alpha_max=False):

    L *= 2

    torch.random.manual_seed(0)
    V = 100

    p_cost = 0.2
    r_cost = 1.
    alpha = 0.5

    y_tgt = torch.randint(4, V + 4, (B, L))
    y_tgt[:, 0] = 0
    y_tgt[:, -1] = 2
    # print(y_tgt)

    

    # lens = torch.randint(2, min(2 + int(L * 0.6), L + 1), (B * N,))
    lens = torch.full((B * N,), min(2 + L // 2, L + 1))

    # mask = torch.arange(L)[None, :].expand(B * N, L) < lens[:, None]
    # mask.view(B, N, L)
    
    idx = torch.arange(L - 2)[None, :].expand(B * N, L - 2)
    shuffler_idx = torch.rand(B * N, L - 2).argsort(-1)
    batch_idx = torch.arange(B * N)[:, None].expand(B * N, L - 2)
    mask_shuffled = torch.ones(B * N, L, dtype=bool)
    mask_shuffled[:, 1:-1] = idx[batch_idx, shuffler_idx] < (lens[:, None] - 2)
    # mask_shuffled[:, 0] = True
    # mask_shuffled[:, -1] = True

    # print("mask_shuffled\n", mask_shuffled)

    # lens_ins = (torch.rand(B * N) * (L - lens + 0.5) + lens).long()
    lens_ins = lens
    mask_ins = torch.arange(L)[None, :].expand(B * N, L) < lens_ins[:, None]

    # print("lens", lens)
    # print("lens_ins", lens_ins)

    y_noise = torch.ones(B * N, L, dtype=torch.long)

    # mask_shuffled_left = ~torch.sort(~mask_shuffled, -1)[0]

    # print("mask shuffled left", mask_shuffled_left)

    idx = torch.arange(1, L - 1)[None, :].expand(B * N, L - 2)
    shuffler = torch.rand(B * N, L - 2)
    shuffler[~mask_ins[:, 2:]] = 1.
    # shuffler[:, 0] = 1.
    # print(torch.arange(B * N)[:, None])
    # print((lens_ins - 1)[None, :])
    # shuffler[torch.arange(B * N), (lens_ins - 1)] = 1.
    # print("shuffler")
    # print(shuffler)
    shuffler_idx = shuffler.sort(dim=-1, descending=False, stable=True)[1]
    # print("shufler idx\n", shuffler_idx)
    batch_idx = torch.arange(B * N)[:, None].expand(B * N, L - 2)

    # print("sorted\n", idx[batch_idx, shuffler_idx])

    mask_shuffled_ins = torch.zeros(B * N, L, dtype=bool)

    mask_shuffled_ins[:, 1:-1] = idx[batch_idx, shuffler_idx] < (lens[:, None] - 1)

    # print("mask_shuffled_ins\n", mask_shuffled_ins)
    mask_shuffled_ins[:, 0] = True
    mask_shuffled_ins[torch.arange(B * N), (lens_ins - 1)] = True
    
    # mask_shuffled_ins = idx[batch_idx, shuffler_idx] < (lens[:, None])

    # print("mask_shuffled_ins\n", mask_shuffled_ins)

    y_tgt_expanded = y_tgt[:, None, :].expand(B, N, L).reshape(B * N, L)
    # print("y init\n", y_tgt_expanded.view(B, N, L))


    y_noise[mask_shuffled_ins] = y_tgt_expanded[mask_shuffled]

    mask_ins_rd = mask_ins & ~mask_shuffled_ins

    # print(y_noise)

    y_noise[mask_ins_rd] = torch.randint(4, V + 4, (mask_ins_rd.sum().item(), ))
    y_noise[~mask_ins] = 1

    x = y_noise.view(B, N, L).clone()

    # print("mask ins\n", mask_ins.long())

    # print(y_tgt)
    # print(x)

    # print(mask_shuffled)
    cum = (~mask_shuffled).cumsum(-1)
    # print(cum)
    delta = cum[mask_shuffled][1:] - cum[mask_shuffled][:-1]
    delta = delta[delta >= 0]

    # print(delta)

    logit_idx = torch.zeros(B, N, L - 1, dtype=torch.long)
    mask_logits = x[:, :, :-1].ne(1) & x[:, :, :-1].ne(2)
    logit_idx[mask_logits] = delta

    # print(logit_idx)

    noise_strength = 0.6
    var_strength = 1.0
    min_var = 0.2

    num_logits = mask_logits.sum()
    mu = (delta.float() + torch.randn_like(delta.float()) * noise_strength)

    # print(delta)
    # print(mu)

    var = min_var + torch.rand_like(delta.float()) * var_strength

    # print(var)

    ks = torch.arange(Kmax + 1, dtype=torch.float)

    # print(ks[None, :] - mu[:, None 

    logits = torch.zeros(B, N, L - 1, Kmax + 1)

    logits[mask_logits] = -(ks[None, :] - mu[:, None]) ** 2 / var[:, None]
    logits[mask_logits] -= torch.logsumexp(logits[mask_logits], -1)[:, None]

    # print(logits)

    # print(logits[mask_logits])

    # print(torch.exp(logits[mask_logits]))

    # print(x.shape, logits.shape)

    #############################
    t1 = time.time()
    realigner = realigner_module.RealignBatch(
        x,
        logits,
        p_cost,
        r_cost,
        alpha,
        2,
        M,
        use_alpha_max
    )
    Dt = time.time() - t1

    print("*", end="", flush=True)

    return Dt


if __name__ == "__main__":
    # print("------- TEST 1 -------")
    # print("Kmax=64, M=65, use_alpha_max=False")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=65, use_alpha_max=False) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST 2 -------")
    # print("Kmax=64, M=65, use_alpha_max=True")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=65, use_alpha_max=True) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST 3 -------")
    # print("Kmax=64, M=16, use_alpha_max=False")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=16, use_alpha_max=False) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST 4 -------")
    # print("Kmax=64, M=4, use_alpha_max=False")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=4, use_alpha_max=False) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST 5 -------")
    # print("Kmax=64, M=2, use_alpha_max=False")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=2, use_alpha_max=False) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST 6 -------")
    # print("Kmax=64, M=2, use_alpha_max=True")
    # N_TEST = 4
    # mDt = sum([test_1(Kmax=64, M=2, use_alpha_max=True) for _ in range(N_TEST)]) / N_TEST
    # print()
    # print("Dt = ", mDt)

    # print("------- TEST B -------")
    # N_TEST = 5
    # for B in [1, 2, 4, 8, 16, 32, 64]:
    #     print(f"B={B}, N=3, L=8, Kmax=64, M=5, use_alpha_max=False")
    #     mDt = sum([test_2(B=B, N=3, L=8, Kmax=64, M=5, use_alpha_max=True) for _ in range(N_TEST)]) / N_TEST
    #     print()
    #     print("Dt = ", mDt)

    # print("------- TEST N -------")
    # N_TEST = 5
    # B = 16
    # L = 8
    # M = 3
    # use_alpha_max = False
    # for N in [2, 3, 4]:
    #     print(f"B={16}, N={3}, L={L}, Kmax=64, M={M}, use_alpha_max={use_alpha_max}")
    #     mDt = sum([test_2(B=B, N=N, L=L, Kmax=64, M=M, use_alpha_max=use_alpha_max) for _ in range(N_TEST)]) / N_TEST
    #     print()
    #     print("Dt = ", mDt)

    # print("------- TEST L -------")
    # N_TEST = 3
    # B = 16
    # N = 3
    # M = 1
    # use_alpha_max = False
    # for L in [64]:
    #     print(f"B={16}, N={3}, L={L}, Kmax=64, M={M}, use_alpha_max={use_alpha_max}")
    #     mDt = sum([test_2(B=B, N=N, L=L, Kmax=64, M=M, use_alpha_max=use_alpha_max) for _ in range(N_TEST)]) / N_TEST
    #     print()
    #     print("Dt = ", mDt)

    print("------- TEST L -------")
    N_TEST = 1
    B = 1
    N = 3
    M = 5
    use_alpha_max = False
    for L in [4, 8, 16, 32, 64]:
        print(f"B={16}, N={3}, L={L}, Kmax=64, M={M}, use_alpha_max={use_alpha_max}")
        mDt = sum([test_3(B=B, N=N, L=L, Kmax=64, M=M, use_alpha_max=use_alpha_max) for _ in range(N_TEST)]) / N_TEST
        print()
        print("Dt = ", mDt)

    

