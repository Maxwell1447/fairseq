# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from fairseq.utils import new_arange
import networkx as nx
import itertools
import sortednp as snp
import numpy as np
import random as rd
from fairseq import libnat2

# -------------- Helper Functions --------------------------------------------------- #


def load_libnat():
    try:
        from fairseq import libnat_cuda

        return libnat_cuda, True

    except ImportError as e:
        print(str(e) + "... fall back to CPU version")

        try:
            from fairseq import libnat

            return libnat, False

        except ImportError as e:
            import sys

            sys.stderr.write(
                "ERROR: missing libnat_cuda. run `python setup.py build_ext --inplace`\n"
            )
            raise e


# def pi_del_(
#     shape,
#     y_tgt_star,
#     pad_symbol=0,
#     plh_symbol=0,
#     bos_symbol=0,
#     eos_symbol=0,
#     Kmax=100,
#     device="cpu",
# ):
#     # shape = B x N x M
#     # y_tgt_star : B x M
#     shape = list(shape)
#     shape[-1] = y_tgt_star.size(-1)
#     shape = tuple(shape)

#     del_tgt = torch.ones(shape, dtype=torch.long, device=device)
#     plh_tgt = -torch.ones(
#         (shape[0], shape[1], shape[2] - 1), dtype=torch.long, device=device
#     )
#     cmb_tgt = -torch.ones(shape[0], shape[2], shape[1], dtype=torch.long, device=device)

#     y_plh = torch.full(
#         (shape[0], shape[1], shape[2]), pad_symbol, dtype=torch.long, device=device
#     )
#     y_cmb = torch.full(shape, pad_symbol, dtype=torch.long, device=device)
#     y_tok = torch.full_like(y_tgt_star, pad_symbol, dtype=torch.long, device=device)

#     tok_mask = torch.zeros_like(y_tgt_star, dtype=bool, device=device)

#     for b, y in enumerate(y_tgt_star):
#         for n in range(shape[1]):

#             mask = (
#                 ((torch.rand(y.shape, device=device) > 0.2) & (y.ne(pad_symbol)))
#                 | (y == bos_symbol)
#                 | (y == eos_symbol)
#             )
#             print("mask", mask)
#             tok_mask[b] = tok_mask[b] + mask
#             y_plh[b, n, : mask.sum()] = y[mask]
#             y_cmb[b, n][y.ne(pad_symbol)] = plh_symbol
#             y_cmb[b, n][mask] = y[mask]

#             idx_tgt = np.arange(shape[2])[mask.cpu().numpy()]
#             print("idx_tgt", idx_tgt)

#             plh_nums = (
#                 np.convolve(
#                     np.concatenate([[-1], idx_tgt, [shape[2]]]), [1, -1], mode="valid"
#                 )
#                 - 1
#             ).clip(0, Kmax - 1)
#             # plh_nums[:-1] = plh_nums[1:]
#             # plh_nums[-1] = 0
#             plh_nums = plh_nums[1:-1]
#             plh_tgt[b, n, : len(plh_nums)] = torch.from_numpy(
#                 plh_nums[: plh_tgt.size(-1)]
#             ).to(device)
#             cmb_tgt[b, : y_tgt_star[b, :].ne(pad_symbol).sum(), n] = 0
#             cmb_tgt[b, idx_tgt, n] = 1
#         y_tok[b] = y
#         y_tok[b][y.ne(pad_symbol) & ~tok_mask[b]] = plh_symbol

#     plh_mask = y_plh.ne(pad_symbol)[:, :, 1:]
#     del_mask = torch.zeros(shape, dtype=bool, device=device)
#     cmb_mask = cmb_tgt.ne(-1)

#     return {
#         "del_tgt": del_tgt,
#         "plh_tgt": plh_tgt,
#         "cmb_tgt": cmb_tgt,
#         "tok_tgt": y_tgt_star,
#         "del_mask": del_mask,
#         "plh_mask": plh_mask,
#         "cmb_mask": cmb_mask,
#         "tok_mask": tok_mask,
#         "y_plh": y_plh,
#         "y_cmb": y_cmb,
#         "y_tok": y_tok,
#     }


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
    sorted_mask = mask.sort(-1, descending=True)[0]
    y_plh[sorted_mask] = y_star_n[mask]
    y_cmb[y_star_n.ne(pad_symbol)] = plh_symbol
    y_cmb[mask] = y_star_n[mask]
    y_tok[y_tgt_star.ne(pad_symbol)] = plh_symbol
    y_tok[tok_mask] = y_tgt_star[tok_mask]

    idx = mask.sort(-1, descending=True)[1]

    plh_tgt = idx[:, :, 1:] - idx[:, :, :-1] - 1
    plh_tgt[~sorted_mask[:, :, 1:]] = 0
    plh_tgt = plh_tgt.clamp(0, Kmax)

    cmb_tgt = mask.long()

    plh_mask = y_plh.ne(pad_symbol)[:, :, 1:]
    del_mask = torch.zeros(shape, dtype=bool, device=device)
    cmb_mask = y_tgt_star.ne(pad_symbol)

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
    # y_cmb_star : B x N x M
    # y_refs : B x N x M

    mask = (y_cmb_star == plh_symbol) * (
        torch.rand(y_cmb_star.shape, device=device) < gamma
    )

    y_cmb = y_cmb_star.clone()
    mask_ref_sel = y_refs.ne(pad_symbol) & y_refs.ne(bos_symbol) & y_refs.ne(eos_symbol)

    dividend = (
        mask_ref_sel.long().sum(-1).unsqueeze(-1).expand(y_refs.shape)
    )  # B x N x M
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


# def pi_star_(
#     y_del, y_star, k=10, pad_symbol=None, plh_symbol=None, Kmax=100, device="cpu"
# ):
#     # y_del : B x N x M
#     # y_star : B x M

#     del_tgt = torch.zeros_like(y_del, dtype=torch.long, device=device)
#     plh_tgt = -torch.ones(
#         y_del.size(0), y_del.size(1), y_del.size(2) - 1, dtype=torch.long, device=device
#     )
#     cmb_tgt = -torch.ones(
#         y_del.size(0), y_del.size(2), y_del.size(1), dtype=torch.long, device=device
#     )

#     y_plh = torch.full_like(y_del, pad_symbol, dtype=torch.long, device=device)
#     y_cmb = torch.full_like(y_del, pad_symbol, dtype=torch.long, device=device)
#     y_tok = torch.full(
#         (y_del.size(0), y_del.size(2)), pad_symbol, dtype=torch.long, device=device
#     )
#     for b in range(y_del.size(0)):
#         ys_retrieved = [y[y.ne(pad_symbol)].cpu().numpy() for y in y_del[b]]
#         y_ref = y_star[b][y_star[b].ne(pad_symbol)].cpu().numpy()
#         best_graph, _ = get_k_best_max_coverage(ys_retrieved, y_ref, k)
#         mask_plh_tok = np.full((y_star.size(1),), False, dtype=bool)
#         mask_y_tok = y_star[b].ne(pad_symbol)
#         for i in range(y_del.size(1)):
#             idx_keep = np.sort(np.array([e[0] for e in best_graph[i]]))
#             idx_tgt = np.sort(np.array([e[1] for e in best_graph[i]], dtype=np.int64))
#             mask_plh_tok[idx_tgt] = True
#             plh_nums = (
#                 np.convolve(
#                     np.concatenate([[-1], idx_tgt, [y_del.size(2)]]),
#                     [1, -1],
#                     mode="valid",
#                 )
#                 - 1
#             ).clip(0, Kmax - 1)
#             # plh_nums[:-1] = plh_nums[1:]
#             # plh_nums[-1] = 0
#             plh_nums = plh_nums[1:-1]
#             plh_tgt[b, i, : len(plh_nums)] = torch.from_numpy(
#                 plh_nums[: plh_tgt.size(-1)]
#             ).to(device)
#             del_tgt[b, i, idx_keep] = 1
#             cmb_tgt[b, : y_star[b, :].ne(pad_symbol).sum(), i] = 0
#             for e in best_graph[i]:
#                 cmb_tgt[b, e[1], i] = 1
#             y_plh[b, i, : len(idx_keep)] = y_del[b, i, idx_keep]
#             y_cmb[b, i, : y_star[b].ne(pad_symbol).sum()] = plh_symbol
#             y_cmb[b, i, idx_tgt] = y_star[b, idx_tgt]
#         y_tok[b][mask_y_tok] = plh_symbol
#         y_tok[b][mask_plh_tok] = y_star[b][mask_plh_tok]

#     del_mask = y_del.ne(pad_symbol)
#     plh_mask = plh_tgt.ne(-1)
#     cmb_mask = cmb_tgt.ne(-1)
#     # cmb_mask = cmb_mask_.any(-1).unsqueeze(-1).expand(cmb_mask_.shape)
#     # cmb_tgt[!cmb_mask_ || cmb_mask] =
#     tok_mask = (y_tok == plh_symbol) & y_star.ne(pad_symbol)

#     return {
#         "del_tgt": del_tgt,
#         "plh_tgt": plh_tgt,
#         "cmb_tgt": cmb_tgt,
#         "tok_tgt": y_star,
#         "del_mask": del_mask,
#         "plh_mask": plh_mask,
#         "cmb_mask": cmb_mask,
#         "tok_mask": tok_mask,
#         "y_plh": y_plh,
#         "y_cmb": y_cmb,
#         "y_tok": y_tok,
#     }


def pi_star(y_del, y_star, k=10, pad_symbol=None, plh_symbol=None, Kmax=100):
    # y_del : B x N x M
    # y_star : B x M
    ops = libnat2.MultiLevEditOps(y_del, y_star, k, pad_symbol, plh_symbol)

    return {
        "del_tgt": ops.get_del(),
        "plh_tgt": ops.get_ins().clamp(Kmax),
        "cmb_tgt": ops.get_cmb(),
        "tok_tgt": y_star,
        "del_mask": y_del.ne(pad_symbol),
        "plh_mask": ops.get_s_ins().ne(pad_symbol),
        "cmb_mask": ops.get_s_cmb()
        .ne(pad_symbol)
        .view(y_star.size(0), y_star.size(1), 1)
        .expand_as(ops.get_s_cmb()),
        "tok_mask": (ops.get_s_ins() == plh_symbol),
        "y_plh": ops.get_s_del(),
        "y_cmb": ops.get_s_ins(),
        "y_tok": ops.get_s_cmb(),
    }


# def to_one_hot(y, voc_size=32000):
#    return torch.nn.functional.one_hot(y, num_classes=voc_size)


# def to_ordinal(y):
#    return torch.argmax(y, -1)


# def smart_build_graph(s_hyp, s_ref):
#    graph = list()
#    left = dict()
#    right = dict()
#    for i, ch in enumerate(s_hyp):
#        if not ch in left:
#            left[ch] = list()
#            right[ch] = list()
#        left[ch].append(i)
#    for j, ch in enumerate(s_ref):
#        if not ch in right:
#            right[ch] = list()
#            if not ch in left:
#                left[ch] = list()
#        right[ch].append(j)
#    for ch in left:
#        for i, j in itertools.product(left[ch], right[ch]):
#            graph.append((i, j))
#    return graph


# def get_coverage_score(coverage):
#    cover = set()
#    for hyp_graph in coverage:
#        for edge in hyp_graph:
#            cover.add(edge[1])
#    return len(cover)

# def get_k_best_from_previous(best_weights, k):
##    print(best_weights)
#    merged = np.array([])
#    merged_keys = np.array([], dtype=object)
#    for key in best_weights:
#        merged, (idx_old, idx_added) = snp.merge(merged, best_weights[key],
#                                                 indices=True)
##        print('merged', merged)
#        new_merged_keys = np.empty_like(merged, dtype=object)
#        new_merged_keys[idx_old] = merged_keys
#        new_merged_keys[idx_added] = key
#        merged_keys = new_merged_keys
#    return merged_keys[-k:], merged[-k:]


# def recursive_backward(G, node, count, path, source):
#    if node == source:
#        return [path]
#    ancestors, _, counts = np.unique(G[node]['k_best_ancestors'][-count:],
#                                       return_counts=True, return_index=True)
##    print('ancestors', ancestors)
#    paths = list()
#    new_path = path if node == 't' else path + [node]
#    for ancestor, count in zip(ancestors, counts):
#        paths += recursive_backward(G, ancestor, count, new_path, source)
#    return paths


# def backward_k_best(G, source, target, k):
#    return recursive_backward(G, target, k, [], source)


# def k_longest_paths(G, source, target, k=1, weight='weight'):
#    data = {v: dict() for v in G.nodes()}
#    data[source]['k_best_ancestors'] = np.array([])
#    data[source]['k_best_weights'] = np.array([0.])
#    for node in nx.topological_sort(G):
#        if node == 's':
#            continue
#        precs = G.predecessors(node)
#        best_weights = {prec: data[prec]['k_best_weights'] + 1 for prec in precs}
#        ancestors, weights = get_k_best_from_previous(best_weights, k)
#        data[node]['k_best_ancestors'] = ancestors
#        data[node]['k_best_weights'] = weights
#    return backward_k_best(data, source, target, k)


# def build_edge_graph(sorted_graph):

#    G = nx.DiGraph()

#    G.add_node('s')
#    G.add_node('t')

#    mapper = {i: sorted_graph[i] for i in range(len(sorted_graph))}

#    G.add_nodes_from(mapper)

#    for i in range(len(sorted_graph)):
#        arc_ref = mapper[i]
#        for j in range(len(sorted_graph) - 1, i, -1):
#            arc = mapper[j]
#            if arc[0] > arc_ref[0] and arc[1] > arc_ref[1]:
#                G.add_edge(i, j, weight=-1)
#    for node in G.nodes():
#        if node in ('s', 't'):
#            continue
#        if G.in_degree(node) == 0:
#            G.add_edge('s', node, weight=-1)
#        if G.out_degree(node) == 0:
#            G.add_edge(node, 't', weight=0)

#    return G


# def sort_graph(graph):
#    order = np.argsort(np.array([sum(e) for e in graph]))
#    return np.array(graph, dtype=tuple)[order]


# def get_k_best_graphs(s_hyp, s_ref, k):
#    graph = smart_build_graph(s_hyp, s_ref)
##    print("full graph", graph)
#    if not graph:
#        return [list()]
#    sorted_graph = sort_graph(graph)
#    G = build_edge_graph(sorted_graph)
#    paths = k_longest_paths(G, 's', 't', k=k, weight='weight')
#    kept = list()
#    cover_sets = list()
#    for path in paths[::-1]:
#        cover = {sorted_graph[p][1] for p in path}
#        for p in cover_sets:
#            if cover.issubset(p):
#                break
#        else:
#            cover_sets.append(cover)
#            kept.append(path)
#    return [[tuple(sorted_graph[i]) for i in path] for path in kept]


# def get_k_best_max_coverage(s_hyps, s_ref, k):
#    hyp_graphs = [get_k_best_graphs(s_hyp, s_ref, k) for s_hyp in s_hyps]
#    max_coverage = None
#    max_coverage_score = -1
##    print(np.prod([len(gs) for gs in hyp_graphs]))
#    if len(hyp_graphs) == 0:
#        return [[] for _ in s_hyps], 0
#    for proposed_coverage in itertools.product(*hyp_graphs):
#        current_score = get_coverage_score(proposed_coverage)
#        if current_score > max_coverage_score:
#            max_coverage_score = current_score
#            max_coverage = proposed_coverage
#    return max_coverage, max_coverage_score


# def make_flat(ys, dim=1):
#    lens = [len(y) for y in ys]
#    return torch.cat(ys, dim=dim), lens

# def make_unflat(y, lens, dim=1):
#    ys = list()
#    cum_len = 0
#    for length in lens:
#        ys.append(y[:, cum_len:length, :])
#        cum_len += length
#    return ys


def _apply_del(in_tokens, word_del_pred, padding_idx, bos_idx, eos_idx):
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


def _apply_plh(in_tokens, plh_pred, padding_idx, unk_idx, eos_idx):
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


def _apply_cmb(in_tokens, cmb_pred, padding_idx, bos_idx, eos_idx, unk_idx):
    # combine choice
    # cmb_pred: B x M x N in [0, 1]
    # in_tokens: B x N x M
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


def _apply_tok(in_tokens, tok_pred, unk_idx):
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
