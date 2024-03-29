# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# run `python setup.py build_ext --inplace` for libnat libraries installation 

import math
import torch
from fairseq.utils import new_arange
from fairseq import libnat2, libnat3
from fairseq import realigner as realigner_module
from fairseq import dist_realign_cuda as dist_realign_cuda

import time
import sys


def pi_del(
    shape,
    y_tgt_star,
    pad_symbol=0,
    plh_symbol=0,
    bos_symbol=0,
    eos_symbol=0,
    Kmax=100,
    mode="binomial",
    device="cpu",
):
    """Operations and states to edit a partially deleted version of y_star back to y_star."""
    # shape = B x N x M    ou B x L
    # y_tgt_star : B x M   ou B x L
    shape = list(shape)
    shape[-1] = y_tgt_star.size(-1)
    shape = tuple(shape)

    del_tgt = torch.ones(shape, dtype=torch.long, device=device)
    plh_tgt = -torch.ones(
        (shape[0], shape[1], shape[2] - 1), dtype=torch.long, device=device
    )
    cmb_tgt = -torch.ones(shape[0], shape[2], shape[1], dtype=torch.long, device=device)

    y_plh = torch.full(
        (shape[0], shape[1], shape[2]), pad_symbol, dtype=torch.long, device=device
    )
    y_cmb = torch.full(shape, pad_symbol, dtype=torch.long, device=device)
    y_tok = torch.full_like(y_tgt_star, pad_symbol, dtype=torch.long, device=device)

    y_star_n = y_tgt_star.view(shape[0], 1, shape[-1]).expand(shape)

    if mode == "uniform":
        raise NotImplementedError(f"{mode} not implemented")
        ...
    else:
        mask = (
            ((torch.rand(y_star_n.shape, device=device) > 0.2) & (y_star_n.ne(pad_symbol)))
            | (y_star_n == bos_symbol)
            | (y_star_n == eos_symbol)
        )

    tok_mask = mask.any(1)
    sorted_ = mask.long().sort(stable=True, descending=True, dim=-1)
    sorted_mask = sorted_[0].bool()
    y_plh[sorted_mask] = y_star_n[mask]
    y_cmb[y_star_n.ne(pad_symbol)] = plh_symbol
    y_cmb[mask] = y_star_n[mask]
    y_tok[y_tgt_star.ne(pad_symbol)] = plh_symbol
    y_tok[tok_mask] = y_tgt_star[tok_mask]

    idx = sorted_[1]

    plh_tgt = idx[:, :, 1:] - idx[:, :, :-1] - 1
    plh_tgt[~sorted_mask[:, :, 1:]] = 0
    plh_tgt = plh_tgt.clamp(0, Kmax - 1)

    cmb_tgt = mask.long()

    plh_mask = y_plh.ne(pad_symbol)[:, :, 1:]
    del_mask = torch.zeros(shape, dtype=bool, device=device)
    cmb_mask = y_tgt_star.ne(pad_symbol).view(shape[0], 1, shape[-1]).expand_as(y_cmb)

    return {
        "del_tgt": del_tgt,
        "plh_tgt": plh_tgt,
        "cmb_tgt": cmb_tgt,
        "tok_tgt": y_tgt_star,
        "del_mask": del_mask,
        "plh_mask": plh_mask,
        "cmb_mask": cmb_mask,
        "tok_mask": tok_mask,
        "y_plh": y_plh,
        "y_cmb": y_cmb,
        "y_tok": y_tok,
    }


def pi_del_single(
    # input_len,
    y_tgt_star,
    pad_symbol=0,
    plh_symbol=0,
    bos_symbol=0,
    eos_symbol=0,
    Kmax=100,
    mode="uniform_length",
    device="cpu",
):
    """Operations and states to edit a partially deleted version of y_star back to y_star."""
    # y_tgt_star : B x M
    shape = list(y_tgt_star.shape)

    plh_tgt = -torch.ones(
        (shape[0], shape[1] - 1), dtype=torch.long, device=device
    )
    y_plh = torch.full(
        (shape[0], shape[1]), pad_symbol, dtype=torch.long, device=device
    )

    if mode == "uniform_length":
        tgt_mask = y_tgt_star.eq(pad_symbol)
        lengths = y_tgt_star.ne(pad_symbol).sum(-1)
        score_select = y_tgt_star.clone().float().uniform_()
        score_select.masked_fill_(
            y_tgt_star.eq(eos_symbol) | y_tgt_star.eq(bos_symbol),
            0.
        )
        score_select.masked_fill_(
            tgt_mask,
            1.
        )
        cutoff = 2 + ((lengths - 2).unsqueeze(1) * score_select.new_zeros(y_tgt_star.size(0), 1).uniform_()).long()
        mask_index = torch.arange(shape[1], dtype=torch.long, device=device)[None, :].expand_as(y_tgt_star) < cutoff
        indexes = score_select.sort(dim=1, stable=True)[1]
        indexes[~mask_index] = 0
        mask = torch.zeros_like(tgt_mask)
        batch_index = torch.arange(shape[0], dtype=torch.long, device=device)[:, None].expand_as(y_tgt_star)
        mask[batch_index, indexes] = True
    else:
        # mask of what is kept
        mask = (
            ((torch.rand(y_tgt_star.shape, device=device) > 0.2) & (y_tgt_star.ne(pad_symbol)))
            | (y_tgt_star == bos_symbol)
            | (y_tgt_star == eos_symbol)
        )

    sorted_ = mask.long().sort(stable=True, descending=True, dim=-1)
    sorted_mask = sorted_[0].bool()
    y_plh[sorted_mask] = y_tgt_star[mask]

    idx = sorted_[1]

    plh_tgt = idx[:, 1:] - idx[:, :-1] - 1
    plh_tgt[~sorted_mask[:, 1:]] = 0
    plh_tgt = plh_tgt.clamp(0, Kmax - 1)

    plh_mask = y_plh.ne(pad_symbol)[:, 1:]


    return {
        "plh_tgt": plh_tgt,
        "plh_mask": plh_mask,
        "y_plh": y_plh,
    }


def pi_sel(
    y_cmb_star,
    y_refs,
    gamma,
    pad_symbol=None,
    plh_symbol=None,
    bos_symbol=None,
    eos_symbol=None,
    device="cuda:0",
):
    """Replace some <plh> by tokens from y_refs (usually the tokens to edit)."""
    # y_cmb_star : B x N x M
    # y_refs : B x N x M

    assert y_cmb_star.shape == y_refs.shape, str(y_cmb_star.shape) + str(y_refs.shape)
    assert ((y_cmb_star == eos_symbol).sum(-1) == 1).all().item(), ((y_cmb_star == bos_symbol).sum(-1) == 1).all().item()

    mask = (y_cmb_star == plh_symbol) * (
        torch.rand(y_cmb_star.shape, device=device) < gamma
    )
    y_cmb = y_cmb_star.clone()
    mask_ref_sel = y_refs.ne(pad_symbol) & y_refs.ne(bos_symbol) & y_refs.ne(eos_symbol)
    dividend = mask_ref_sel.sum(-1).unsqueeze(-1).expand(y_refs.shape)  # B x N x M
    mask_void = (dividend[:, :, 0].ne(0).all(-1))
    idxs = new_arange(y_refs[mask_void])

    idxs = torch.remainder(idxs, dividend[mask_void]) + 1
    idxs = idxs[:, :, torch.randperm(idxs.size(-1))]
    mask[~mask_void] = False

    y_cmb[mask] = torch.gather(y_refs[mask_void], 2, idxs)[mask[mask_void]]

    return y_cmb


def pi_mask(
    y_star,
    pad_symbol=None,
    plh_symbol=None,
    bos_symbol=None,
    eos_symbol=None,
    device="cuda:0",
):
    """Mask some tokens with <plh> from the target sequence and learn to predict correct tokens."""
    y_tok = y_star.clone()

    y_tok[
        (torch.rand(y_tok.shape, device=device) > 0.7)
        * (y_tok.ne(pad_symbol))
        * y_tok.ne(bos_symbol)
        * y_tok.ne(eos_symbol)
    ] = plh_symbol
    tok_mask = y_tok == plh_symbol
    tok_tgt = y_star

    return y_tok, tok_tgt, tok_mask


def pi_star(
    y_del, y_star, idf_tgt=None, k=10, max_valency=-1, pad_symbol=None, plh_symbol=None, Kmax=100, device="cuda:0"
):
    """Quasi optimal operations and states to edit y_del to y_star"""
    # y_del : B x N x M
    # y_star : B x M
    if y_del.size(1) == 1:
        k = 1
    if idf_tgt is not None:
        print("exs", y_del.shape, y_del.device, y_del.dtype, y_del, file=sys.stderr)
        print("tgt", y_star.shape, y_star.device, y_star.dtype, y_star, file=sys.stderr)
        print("idf", idf_tgt.shape, idf_tgt.device, idf_tgt.dtype, idf_tgt, file=sys.stderr)
        ops = libnat3.MultiLevEditOpsIDF(
            y_del.cpu(), y_star.cpu(), idf_tgt.cpu().contiguous().float(),
            k, max_valency, 0.1,
            pad_symbol, plh_symbol)
        # ops2 = libnat2.MultiLevEditOps(y_del.cpu(), y_star.cpu(), k, max_valency, pad_symbol, plh_symbol)
    else:
        ops = libnat2.MultiLevEditOps(y_del.cpu(), y_star.cpu(), k, max_valency, pad_symbol, plh_symbol)

    cmb_tgt = ops.get_cmb().to(device)
    y_tok = ops.get_s_cmb().to(device)
    

    return {
        "del_tgt": ops.get_del().to(device),
        "plh_tgt": ops.get_ins().clamp(0, Kmax - 1).to(device),
        "cmb_tgt": cmb_tgt,
        "tok_tgt": y_star,
        "del_mask": y_del.ne(pad_symbol),
        "plh_mask": ops.get_s_del().ne(pad_symbol).to(device)[:, :, 1:],
        "cmb_mask": y_star.ne(pad_symbol)
            .view(y_star.size(0), 1, y_star.size(1))
            .expand_as(ops.get_s_ins()),
        "tok_mask": (ops.get_s_cmb().to(device) == plh_symbol),
        "y_plh": ops.get_s_del().to(device),
        "y_cmb": ops.get_s_ins().to(device),
        "y_tok": y_tok,
    }

    # print("y_plh 2", ops2.get_s_del().to(device), file=sys.stderr)
    # print("y_plh 3", res["y_plh"], file=sys.stderr)


    # cmb_tgt = ops2.get_cmb().to(device)
    # y_tok = ops2.get_s_cmb().to(device)
    
    # res2 = {
    #     "del_tgt": ops2.get_del().to(device),
    #     "plh_tgt": ops2.get_ins().clamp(0, Kmax - 1).to(device),
    #     "cmb_tgt": cmb_tgt,
    #     "tok_tgt": y_star,
    #     "del_mask": y_del.ne(pad_symbol),
    #     "plh_mask": ops2.get_s_del().ne(pad_symbol).to(device)[:, :, 1:],
    #     "cmb_mask": y_star.ne(pad_symbol)
    #         .view(y_star.size(0), 1, y_star.size(1))
    #         .expand_as(ops2.get_s_ins()),
    #     "tok_mask": (ops2.get_s_cmb().to(device) == plh_symbol),
    #     "y_plh": ops2.get_s_del().to(device),
    #     "y_cmb": ops2.get_s_ins().to(device),
    #     "y_tok": y_tok,
    # }
    # print("res", res, file=sys.stderr)
    # print("res2", res2, file=sys.stderr)
    # sys.exit()
    
    # return res

def handle_all_plh_case(cmb_tgt, y_tok, y_cmb, plh_symbol):
    # if position with only plh, consider them as acceptable, but only if plh
    # msk_cmb_sel = ((y_tok == plh_symbol) & ((y_cmb == plh_symbol).all(1))).unsqueeze(1).expand_as(cmb_tgt) & (y_cmb == plh_symbol)
    msk_cmb_sel = ((y_tok == plh_symbol) & (~cmb_tgt.any(1))).unsqueeze(1).expand_as(cmb_tgt) & (y_cmb == plh_symbol)
    cmb_tgt[msk_cmb_sel] = 1
    return cmb_tgt


def apply_del(in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx, in_origin=None):
    # word_del_pred: B x N x M in {False, True}
    # apply deletion to a tensor
    in_masks = in_tokens.ne(padding_idx)
    bos_eos_masks = in_tokens.eq(bos_idx) | in_tokens.eq(eos_idx)

    max_len = in_tokens.size(2)
    word_del_pred = word_del_pred.bool()
    word_del_pred = ~word_del_pred
    word_del_pred.masked_fill_(~in_masks, 1)
    word_del_pred.masked_fill_(bos_eos_masks, 0)  # do not delete bos/eos

    reordering = new_arange(in_tokens).masked_fill_(word_del_pred, max_len).sort(2)[1]

    out_tokens = in_tokens.masked_fill(word_del_pred, padding_idx).gather(2, reordering)

    out_scores = None
    if in_scores is not None:
        out_scores = in_scores.masked_fill(word_del_pred, 0).gather(2, reordering)
    out_origin = None
    if in_origin is not None:
        out_origin = in_origin.masked_fill(word_del_pred, 0).gather(2, reordering)

    out_attn = None
    if in_attn is not None:
        _mask = word_del_pred[:, :, :, None].expand_as(in_attn)
        _reordering = reordering[:, :, :, None].expand_as(in_attn)
        out_attn = in_attn.masked_fill(_mask, 0.0).gather(2, _reordering)

    return out_tokens, out_scores, out_attn, out_origin


def apply_plh(in_tokens, in_scores, plh_pred, padding_idx, unk_idx, eos_idx, to_ignore_mask=None, in_origin=None):
    # plh_pred: B x N x M in {0, 1, ..., K_max - 1}
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(2)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    plh_pred.masked_fill_(~in_masks[:, :, 1:], 0)

    out_lengths = in_lengths + plh_pred.sum(2)  # B x N
    out_masks = (
        new_arange(out_lengths, out_lengths.max())[None, :] < out_lengths[:, :, None]
    )

    reordering = (plh_pred + in_masks[:, :, 1:].long()).cumsum(2)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), in_tokens.size(1), out_lengths.max())
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, :, 0] = in_tokens[:, :, 0]

    out_tokens.scatter_(2, reordering, in_tokens[:, :, 1:])
    out_tokens.scatter_(2, reordering[:, :, -1:], eos_idx)

    if to_ignore_mask is not None:
        idx_n = to_ignore_mask.to(torch.int16).argsort(-1)[:, 0][:, None].expand(out_tokens.size(0), out_tokens.size(1))
        idx_b = torch.arange(out_tokens.size(0), device=out_tokens.device)[:, None].expand_as(idx_n)
        out_tokens[to_ignore_mask] = out_tokens[idx_b[to_ignore_mask], idx_n[to_ignore_mask]]

    out_scores = None
    if in_scores is not None:
        in_scores.masked_fill_(~in_masks, 0)
        out_scores = in_scores.new_zeros(*out_tokens.size())
        out_scores[:, :, 0] = in_scores[:, :, 0]
        out_scores.scatter_(2, reordering, in_scores[:, :, 1:])
        out_scores.scatter_(2, reordering[:, :, -1:], 0)

    out_origin = None
    if in_origin is not None:
        in_origin.masked_fill_(~in_masks, 0)
        out_origin = in_origin.new_zeros(*out_tokens.size())
        out_origin[:, :, 0] = in_origin[:, :, 0]
        out_origin.scatter_(2, reordering, in_origin[:, :, 1:])
        out_origin.scatter_(2, reordering[:, :, -1:], 0)

    return out_tokens, out_scores, out_origin


def apply_cmb(in_tokens, in_scores, cmb_pred, padding_idx, bos_idx, eos_idx, unk_idx, in_origin=None):
    # combine choice
    # cmb_pred: B x M x N in [0, 1] (float!)
    # in_tokens: B x N x M
    lengths = in_tokens.ne(padding_idx).sum(-1)
    cmb_pred[
        (in_tokens == eos_idx).transpose(1, 2) &
        (lengths.ne(lengths.max(-1)[0][..., None]))[..., None, :]
    ] = torch.finfo(cmb_pred.dtype).min
    cmb_pred = cmb_pred.max(-1)[1]
    in_masks = in_tokens.ne(padding_idx)
    in_cmb_lengths = (in_masks.sum(1) > 0).sum(-1)  # B

    out_tokens = torch.full(
        (in_tokens.size(0), in_tokens.size(2)), padding_idx, device=in_tokens.device
    )
    out_masks = (
        new_arange(in_cmb_lengths, in_tokens.size(-1))[None, :]
        < in_cmb_lengths[:, None]
    )

    idx1 = (
        new_arange(in_cmb_lengths, in_tokens.size(0))
        .expand(in_tokens.size(2), in_tokens.size(0))
        .t()
    )
    idx2 = new_arange(in_cmb_lengths, in_tokens.size(2)).expand(
        in_tokens.size(0), in_tokens.size(2)
    )

    chosen = in_tokens.transpose(1, 2)[idx1, idx2, cmb_pred]

    out_tokens[out_masks] = chosen[out_masks]

    out_scores = None
    if in_scores is not None:
        out_scores = torch.full(
            (in_tokens.size(0), in_tokens.size(2)),
            0.,
            device=in_tokens.device,
            dtype=in_scores.dtype
        )
        chosen_score = in_scores.transpose(1, 2)[idx1, idx2, cmb_pred]
        out_scores[out_masks] = chosen_score[out_masks]
    out_origin = None
    if in_origin is not None:
        out_origin = torch.full(
            (in_tokens.size(0), in_tokens.size(2)),
            0.,
            device=in_origin.device,
            dtype=in_origin.dtype
        )
        chosen_origin = in_origin.transpose(1, 2)[idx1, idx2, cmb_pred]
        out_origin[out_masks] = chosen_origin[out_masks]

    return out_tokens, out_scores, out_origin


def apply_tok(in_tokens, in_scores, tok_pred, tok_scores, unk_idx, in_origin=None):
    tok_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(tok_masks, tok_pred[tok_masks])

    if in_scores is not None:
        out_scores = in_scores.masked_scatter(
            tok_masks, tok_scores[tok_masks]
        )
    else:
        out_scores = None
    if in_origin is not None:
        out_origin = in_origin
    else:
        out_origin = None

    return out_tokens, out_scores, out_origin


def realign_dp_malign(
    tokens_to_realign,
    logits,
    p_cost=0.2,
    r_cost=1.0,
    alpha=0.5,
    M=2,
    use_alpha_max=False,
    eos=2,
):
    t1 = time.time()
    print("\n<<<", flush=True, file=sys.stderr)
    realigner = realigner_module.RealignBatch(
        tokens_to_realign.cpu(),
        logits.cpu().float(),
        p_cost,
        r_cost,
        alpha,
        eos,
        M,
        use_alpha_max
    )
    print("\ntime =", time.time() - t1, flush=True, file=sys.stderr)
    return realigner.get_realigned_plh_pred().to(tokens_to_realign.device), realigner.get_success_mask().bool().to(tokens_to_realign.device)


def build_alignment_graph(x, logits, pad, max_dist=2.5):
    init_t = logits.argmax(-1)
    b = init_t.size(0)
    n = init_t.size(1)
    l = init_t.size(2) + 1
    init_pos = torch.zeros(b, n, l, dtype=init_t.dtype, device=init_t.device)
    init_pos[..., 1:] = init_t + 1
    init_pos = init_pos.cumsum(-1).float()

    if max_dist is not None and max_dist < l:
        dist_tensor = torch.cdist(
            init_pos.view(b, -1)[..., None], init_pos.view(b, -1)[..., None], p=1.0
        ).view(b, n, l, n, l)

    graph = dist_realign_cuda.get_graph(x, pad).bool()
    if max_dist is not None and max_dist < l:
        graph = graph & (dist_tensor <= max_dist)

    graph_mask = graph.view(b, n, l, -1).any(-1)
    graph = graph.long()

    return graph, graph_mask


def compute_regression_normal(logits, logits_mask):
    probs = torch.softmax(logits[logits_mask], dim=-1) ** 2
    probs = probs / probs.sum(-1)[..., None]
    arr = torch.arange(probs.size(-1), device=logits.device, dtype=logits.dtype)[
        None, :
    ]
    mu = (probs * arr).sum(-1)
    sigma2 = (probs * (arr**2)).sum(-1) - mu**2
    mu = (mu + 5 * logits[logits_mask].argmax(-1)) / 6
    return mu, sigma2.clamp(0.1, 2.0)


def log_prob_loss_multinomial_pdf(params, logits, sigma=1.0, tau=1.0):
    # params: M
    # logits: M x (Kmax + 1)
    p = torch.exp(tau * (logits - logits.max(-1)[0][:, None]))
    p /= p.sum(-1)[:, None]
    ii = torch.arange(logits.size(-1) + 1, device=p.device, dtype=p.dtype)[None, :]

    # numerically stable log-sum-exp
    xs = - 0.5 * ((params[:, None] - ii) / sigma) ** 2  # M x (Kmax + 1)
    max_stablizer = xs.max(-1)[0]
    ys = torch.log(torch.exp(xs - max_stablizer[:, None]).sum(-1)) # M
    
    return ys.sum()

def log_prob_loss_normal(t, mask_param, mu, sigma2):
    return ((t[mask_param] - mu) ** 2 / (2 * sigma2)).sum()


def length_loss(t, mask_param):
    lengths = mask_param.sum(-1) + t.sum(-1)
    mean_length = lengths.detach().mean(-1)[..., None].expand_as(lengths)
    y = torch.sigmoid((lengths - mean_length) * 4 * 2.0)
    return (1 - 4 * (1 - y) * y).sum()


def integer_loss(t, mask_param):
    return (torch.sin(t[mask_param] * torch.pi) ** 2).sum()


def lambda_t(it, kind="square", start=0, end=100, gamma=1.0):
    if it < start:
        return 0.0
    if it > end:
        return gamma
    if kind == "square":
        g = gamma / (end - start) ** 2
        return g * (it - start) ** 2
    else:
        return 0.0


def align_tok_loss(t, graph, graph_mask, per_batch=False, p=1):
    b = t.size(0)
    n = t.size(1)
    l = t.size(2) + 1
    pos = torch.zeros(b, n, l, dtype=t.dtype, device=t.device)
    pos[..., 1:] = t + 1
    pos = pos.cumsum(-1)

    min_dist_align = dist_realign_cuda.scatter_dist_lp(graph, pos, graph_mask.long(), p)

    min_dist_align[~graph_mask]
    if per_batch:
        min_dist_align[~graph_mask] = 0.0
        min_dist_align = min_dist_align.view(min_dist_align.size(0), -1).sum(-1)
        return min_dist_align / graph_mask.view(graph_mask.size(0), -1).sum(-1)

    return min_dist_align[graph_mask].sum()

"""
optuna_best = {
    'max_dist': 5,
    'lr': 0.005,
    'momentum': 0.97,
    'scheduler_sqrt_rate': 0.5,
    'num_iter': 100,
    'alpha': 0.4,
    'gamma': 0.8,
    'start': 0.65,
    'end': 0.8,
    'len_loss_scale': 0.6
}
"""
def realign_grad_descent(
    x,
    logits,
    Kmax=64,
    bos=0,
    pad=1,
    eos=2,
    unk=3,
    max_dist=5.0,
    lr=0.005,
    momentum=0.97,
    scheduler_sqrt_rate=0.5,
    num_iter=100,
    alpha=0.4,
    gamma=0.8,
    start=0.65,
    end=0.8,
    len_loss_scale=0.6,
    p=2,
    log_prob_loss_type="normal_regression",
    sigma=1.0,
    tau=1.0
):
    mask_param = x[..., 1:].ne(pad)

    if log_prob_loss_type == "normal_regression":
        mu, sigma2 = compute_regression_normal(logits, mask_param)
    elif log_prob_loss_type == "multinomial_pdf":
        mu = logits[mask_param].argmax(-1).to(logits.dtype)

    params_ = mu.clamp(0, Kmax)

    with torch.enable_grad():
        params_.requires_grad_()

        graph, graph_mask = build_alignment_graph(x, logits, pad, max_dist=max_dist)

        ##### OPTIMZER PARAMS
        optimizer = torch.optim.SGD([params_], lr=lr, momentum=momentum)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=(lambda x: 1.0 / math.sqrt(x * scheduler_sqrt_rate + 1.0))
        )

        for it in range(num_iter):
            optimizer.zero_grad()
            params = torch.zeros_like(x[..., :-1], dtype=logits.dtype)
            params[mask_param] = params_
            if log_prob_loss_type == "normal_regression":
                loss_prob = log_prob_loss_normal(params, mask_param, mu, sigma2)
            elif log_prob_loss_type == "multinomial_pdf":
                loss_prob = log_prob_loss_multinomial_pdf(params[mask_param], logits[mask_param], sigma=sigma, tau=tau)
            loss_len = (
                length_loss(params, mask_param)
                if len_loss_scale > 0.0
                else torch.tensor(0.0, device=x.device)
            )
            
            loss_align = align_tok_loss(params, graph, graph_mask, p=p)
            loss = alpha * loss_prob + (1 - alpha) * loss_align + loss_len * len_loss_scale
            int_loss = integer_loss(params, mask_param)

            loss_tot = (
                loss
                + lambda_t(
                    it,
                    start=start * num_iter,
                    end=end * num_iter,
                    gamma=gamma,
                    kind="square",
                )
                * int_loss
            )

            loss_tot.backward()
            
            optimizer.step()
            scheduler.step()
            params_.data = torch.clamp(params_, 0, Kmax)

    params = torch.zeros_like(x[..., :-1], dtype=x.dtype)
    params[mask_param] = params_.detach().round().to(x.dtype)
    return params
    
    

def _skip(x, mask):
    """
    Getting sliced (dim=0) tensor by mask. Supporting tensor and list/dict of tensors.
    """
    if isinstance(x, int):
        return x

    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        if x.size(0) == mask.size(0):
            return x[mask]
        elif x.size(1) == mask.size(0):
            return x[:, mask]

    if isinstance(x, list):
        return [_skip(x_i, mask) for x_i in x]

    if isinstance(x, dict):
        return {k: _skip(v, mask) for k, v in x.items()}

    raise NotImplementedError


def _skip_encoder_out(encoder, encoder_out, mask):
    if not mask.any():
        return encoder_out
    else:
        return encoder.reorder_encoder_out(
            encoder_out, mask.nonzero(as_tuple=False).squeeze()
        )


def _fill(x, mask, y, padding_idx):
    """
    Filling tensor x with y at masked positions (dim=0).
    """
    if x is None:
        return y
    assert x.dim() == y.dim() and mask.size(0) == x.size(0)
    assert x.dim() == 2 or x.dim() == 3 or (x.dim() == 4 and x.size(3) == y.size(3))
    n_selected = mask.sum()
    assert n_selected == y.size(0)

    if n_selected == x.size(0):
        return y

    if x.size(1) < y.size(1):
        dims = [x.size(0), y.size(1) - x.size(1)]
        if x.dim() == 3:
            dims.append(x.size(2))
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 1)
        x[mask] = y
    elif x.size(1) > y.size(1):
        x[mask] = padding_idx
        if x.dim() == 2:
            x[mask, : y.size(1)] = y
        elif x.dim() == 3:
            x[mask, : y.size(1), :] = y
        else:
            x[mask, : y.size(1), :, :] = y
    elif x.dim() == 3 and (x.size(2) < y.size(2)):
        dims = [x.size(0), x.size(1), y.size(2) - x.size(2)]
        x = torch.cat([x, x.new_zeros(*dims).fill_(padding_idx)], 2)
        x[mask] = y
    elif x.dim() == 3 and (x.size(2) > y.size(2)):
        x[mask] = padding_idx
        x[mask, :, :y.size(2)] = y
    else:
        x[mask] = y
    return x
