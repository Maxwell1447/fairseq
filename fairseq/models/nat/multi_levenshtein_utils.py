# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.utils import new_arange

# import networkx as nx
# import itertools
# import sortednp as snp
# import numpy as np
# import random as rd
from fairseq import libnat2

# -------------- Helper Functions --------------------------------------------------- #


# def load_libnat():
#     try:
#         from fairseq import libnat_cuda

#         return libnat_cuda, True

#     except ImportError as e:
#         print(str(e) + "... fall back to CPU version")

#         try:
#             from fairseq import libnat

#             return libnat, False

#         except ImportError as e:
#             import sys

#             sys.stderr.write(
#                 "ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n"
#             )
#             raise e


def pi_del(
    shape,
    y_tgt_star,
    pad_symbol=0,
    plh_symbol=0,
    bos_symbol=0,
    eos_symbol=0,
    Kmax=100,
    device="cpu",
):
    """Operations and states to edit a partially deleted version of y_star back to y_star."""
    # shape = B x N x M
    # y_tgt_star : B x M
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

    # tok_mask = torch.zeros_like(y_star_n, dtype=bool, device=device)
    mask = (
        ((torch.rand(y_star_n.shape, device=device) > 0.2) & (y_star_n.ne(pad_symbol)))
        | (y_star_n == bos_symbol)
        | (y_star_n == eos_symbol)
    )

    tok_mask = mask.any(1)
    sorted_ = mask.long().sort(-1, descending=True)
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

    mask = (y_cmb_star == plh_symbol) * (
        torch.rand(y_cmb_star.shape, device=device) < gamma
    )
    y_cmb = y_cmb_star.clone()
    mask_ref_sel = y_refs.ne(pad_symbol) & y_refs.ne(bos_symbol) & y_refs.ne(eos_symbol)
    dividend = mask_ref_sel.sum(-1).unsqueeze(-1).expand(y_refs.shape)  # B x N x M
    idxs = new_arange(y_refs)
    idxs = torch.remainder(idxs, dividend) + 1
    idxs = idxs[:, :, torch.randperm(idxs.size(-1))]

    y_cmb[mask] = torch.gather(y_refs, 2, idxs)[mask]

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
    y_del, y_star, k=10, pad_symbol=None, plh_symbol=None, Kmax=100, device="cuda:0"
):
    """Quasi optimal operations and states to edit y_del to y_star"""
    # y_del : B x N x M
    # y_star : B x M
    ops = libnat2.MultiLevEditOps(y_del.cpu(), y_star.cpu(), k, pad_symbol, plh_symbol)

    return {
        "del_tgt": ops.get_del().to(device),
        "plh_tgt": ops.get_ins().clamp(0, Kmax - 1).to(device),
        "cmb_tgt": ops.get_cmb().to(device),
        "tok_tgt": y_star,
        "del_mask": y_del.ne(pad_symbol),
        "plh_mask": ops.get_s_ins().ne(pad_symbol).to(device)[:, :, 1:],
        "cmb_mask": y_star.ne(pad_symbol)
        .view(y_star.size(0), 1, y_star.size(1))
        .expand_as(ops.get_s_ins()),
        "tok_mask": (ops.get_s_cmb().to(device) == plh_symbol),
        "y_plh": ops.get_s_del().to(device),
        "y_cmb": ops.get_s_ins().to(device),
        "y_tok": ops.get_s_cmb().to(device),
    }


def apply_del(in_tokens, word_del_pred, padding_idx, bos_idx, eos_idx):
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

    return out_tokens


def apply_plh(in_tokens, plh_pred, padding_idx, unk_idx, eos_idx):
    # plh_pred: B x N x M in {0, 1, ..., K_max - 1}
    in_masks = in_tokens.ne(padding_idx)
    in_lengths = in_masks.sum(2)

    # HACK: hacky way to shift all the paddings to eos first.
    in_tokens.masked_fill_(~in_masks, eos_idx)
    plh_pred.masked_fill_(~in_masks[:, :, 1:], 0)

    out_lengths = in_lengths + plh_pred.sum(2)  # B x N
    out_masks = (
        new_arange(out_lengths, in_tokens.size(2))[None, :] < out_lengths[:, :, None]
    )

    reordering = (plh_pred + in_masks[:, :, 1:].long()).cumsum(2)
    out_tokens = (
        in_tokens.new_zeros(in_tokens.size(0), in_tokens.size(1), in_tokens.size(2))
        .fill_(padding_idx)
        .masked_fill_(out_masks, unk_idx)
    )
    out_tokens[:, :, 0] = in_tokens[:, :, 0]
    out_tokens.scatter_(2, reordering, in_tokens[:, :, 1:])

    return out_tokens


def apply_cmb(in_tokens, cmb_pred, padding_idx, bos_idx, eos_idx, unk_idx):
    # combine choice
    # cmb_pred: B x M x N in [0, 1] (float!)
    # in_tokens: B x N x M
    cmb_pred = cmb_pred.transpose(1, 2).max(-1)[1]
    in_masks = in_tokens.ne(padding_idx)
    in_cmb_lengths = (in_masks.sum(1) > 0).sum(-1)  # B

    out_tokens = torch.full(
        (in_tokens.size(0), in_tokens.size(2)), padding_idx, device=in_tokens.device
    )
    out_masks = (
        new_arange(in_cmb_lengths, in_tokens.size(-1))[None, :]
        < in_cmb_lengths[:, None]
    )
    #    out_tokens[out_masks] = unk_idx

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

    return out_tokens


def apply_tok(in_tokens, tok_pred, unk_idx):
    tok_masks = in_tokens.eq(unk_idx)
    out_tokens = in_tokens.masked_scatter(tok_masks, tok_pred[tok_masks])

    return out_tokens


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
    assert x.dim() == 2 or (x.dim() == 3 and x.size(2) == y.size(2))
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
    else:
        x[mask] = y
    return x
