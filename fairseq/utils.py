# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import copy
import importlib
import logging
import os
import sys
import tempfile
import warnings
from itertools import accumulate
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F
from fairseq.modules.multihead_attention import MultiheadAttention
from torch import Tensor


try:
    from amp_C import multi_tensor_l2norm

    multi_tensor_l2norm_available = True
except ImportError:
    multi_tensor_l2norm_available = False

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


logger = logging.getLogger(__name__)


MANIFOLD_PATH_SEP = "|"


class FileContentsAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(FileContentsAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        from fairseq.file_io import PathManager

        if PathManager.isfile(values):
            with PathManager.open(values) as f:
                argument = f.read().strip()
        else:
            argument = values
        setattr(namespace, self.dest, argument)


def split_paths(paths: str) -> List[str]:
    return (
        paths.split(os.pathsep)
        if "://" not in paths
        else paths.split(MANIFOLD_PATH_SEP)
    )


def load_ensemble_for_inference(filenames, task, model_arg_overrides=None):
    from fairseq import checkpoint_utils

    deprecation_warning(
        "utils.load_ensemble_for_inference is deprecated. "
        "Please use checkpoint_utils.load_model_ensemble instead."
    )
    return checkpoint_utils.load_model_ensemble(
        filenames, arg_overrides=model_arg_overrides, task=task
    )


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


def move_to_cpu(sample):

    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


def move_to_tpu(sample):

    import torch_xla.core.xla_model as xm
    device = xm.xla_device()

    def _move_to_tpu(tensor):
        return tensor.to(device)

    return apply_to_sample(_move_to_tpu, sample)


def get_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
) -> Optional[Dict[str, Optional[Tensor]]]:
    """Helper for getting incremental state for an nn.Module."""
    return module.get_incremental_state(incremental_state, key)


def set_incremental_state(
    module: MultiheadAttention,
    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    key: str,
    value: Dict[str, Optional[Tensor]],
) -> Optional[Dict[str, Dict[str, Optional[Tensor]]]]:
    """Helper for setting incremental state for an nn.Module."""
    if incremental_state is not None:
        result = module.set_incremental_state(incremental_state, key, value)
        if result is not None:
            incremental_state = result
    return incremental_state


def load_align_dict(replace_unk):
    if replace_unk is None:
        align_dict = None
    elif isinstance(replace_unk, str) and len(replace_unk) > 0:
        # Load alignment dictionary for unknown word replacement if it was passed as an argument.
        align_dict = {}
        with open(replace_unk, "r") as f:
            for line in f:
                cols = line.split()
                align_dict[cols[0]] = cols[1]
    else:
        # No alignment dictionary provided but we still want to perform unknown word replacement by copying the
        # original source word.
        align_dict = {}
    return align_dict


def print_embed_overlap(embed_dict, vocab_dict):
    embed_keys = set(embed_dict.keys())
    vocab_keys = set(vocab_dict.symbols)
    overlap = len(embed_keys & vocab_keys)
    logger.info("found {}/{} types in embedding file".format(overlap, len(vocab_dict)))


def parse_embedding(embed_path):
    """Parse embedding text file into a dictionary of word and embedding tensors.

    The first line can have vocabulary size and dimension. The following lines
    should contain word and embedding separated by spaces.

    Example:
        2 5
        the -0.0230 -0.0264  0.0287  0.0171  0.1403
        at -0.0395 -0.1286  0.0275  0.0254 -0.0932
    """
    embed_dict = {}
    with open(embed_path) as f_embed:
        next(f_embed)  # skip header
        for line in f_embed:
            pieces = line.rstrip().split(" ")
            embed_dict[pieces[0]] = torch.Tensor(
                [float(weight) for weight in pieces[1:]]
            )
    return embed_dict


def load_embedding(embed_dict, vocab, embedding):
    for idx in range(len(vocab)):
        token = vocab[idx]
        if token in embed_dict:
            embedding.weight.data[idx] = embed_dict[token]
    return embedding


def replace_unk(hypo_str, src_str, alignment, align_dict, unk):
    from fairseq import tokenizer

    # Tokens are strings here
    hypo_tokens = tokenizer.tokenize_line(hypo_str)
    # TODO: Very rare cases where the replacement is '<eos>' should be handled gracefully
    src_tokens = tokenizer.tokenize_line(src_str) + ["<eos>"]
    for i, ht in enumerate(hypo_tokens):
        if ht == unk:
            src_token = src_tokens[alignment[i]]
            # Either take the corresponding value in the aligned dictionary or just copy the original value.
            hypo_tokens[i] = align_dict.get(src_token, src_token)
    return " ".join(hypo_tokens)


def post_process_prediction(
    hypo_tokens,
    src_str,
    alignment,
    align_dict,
    tgt_dict,
    remove_bpe=None,
    extra_symbols_to_ignore=None,
):
    hypo_str = tgt_dict.string(
        hypo_tokens, remove_bpe, extra_symbols_to_ignore=extra_symbols_to_ignore
    )
    if align_dict is not None:
        hypo_str = replace_unk(
            hypo_str, src_str, alignment, align_dict, tgt_dict.unk_string()
        )
    if align_dict is not None or remove_bpe is not None:
        # Convert back to tokens for evaluating with unk replacement or without BPE
        # Note that the dictionary can be modified inside the method.
        hypo_tokens = tgt_dict.encode_line(hypo_str, add_if_not_exist=True)
    return hypo_tokens, hypo_str, alignment


def write_formatted_ops_and_stages(
    src_str,
    tgt_str,
    hyp_str,
    history_toks,
    history_ops,
    filename,
    tgt_dict,
):
    with open(filename, 'a') as f:
        f.write("<hr>\n")
        f.write("<hr>\n")
        f.write("""<p style="line-height:85%"><b>src: </b>""")
        f.write(src_str)
        f.write("""</p>\n<p style="line-height:85%"><b>tgt: </b>""")
        f.write(tgt_str)
        f.write("""</p><p style="line-height:85%"><b>hyp: </b>\n""")
        f.write(hyp_str)
        f.write("""</p>\n""")
        
    for i in range(len(history_toks) - 1):
        if history_toks[i]["tokens"].dim() > 1:
            toks_list = [tgt_dict.string(
                history_toks[i]["tokens"][j], None, extra_symbols_to_ignore=None
            ).split(" ") for j in range(history_toks[i]["tokens"].size(0))]
        else:
            toks_list = [tgt_dict.string(
                history_toks[i]["tokens"], None, extra_symbols_to_ignore=None
            ).split(" ")]
        iii = i - i // 3
        if i == 0 or (i > 3 and (i % 3 == 1)):
            # del
            for c in range(len(toks_list)):
                for t in range(len(toks_list[c])):
                    if history_ops[iii]["ops"].dim() == 1:
                        op = history_ops[iii]["ops"][t + 1]
                    else:
                        if t + 1 >= len(history_ops[iii]["ops"][c]):
                            continue
                        op = history_ops[iii]["ops"][c][t + 1]
                    if op.item():
                        toks_list[c][t] = """<strike><span style="color:#AA0000">""" + toks_list[c][t] + "</strike></span>"            
        elif i == 1 or (i > 3 and (i % 3 == 2)):
            # plh
            for c in range(len(toks_list)):
                toks_list[c] = [""] + toks_list[c]
                for t in range(len(toks_list[c])):
                    if history_ops[iii]["ops"].dim() == 1:
                        op = history_ops[iii]["ops"][t]
                    else:
                        if t >= len(history_ops[iii]["ops"][c]):
                            continue
                        op = history_ops[iii]["ops"][c][t]
                    if op.item() > 0:
                        toks_list[c][t] = toks_list[c][t] + """<span style="color:#FF0000">+""" + str(op.item())  + "</span>"
        elif i == 2:
            # cmb
            for c in range(len(toks_list)):
                for t in range(len(toks_list[c])):
                    op = history_ops[iii]["ops"][t + 1][c]
                    if toks_list[c][t] == "":
                        toks_list[c][t] = "■"
                    if op.item() > 0.5:
                        toks_list[c][t] = "<u>" + toks_list[c][t] + "</u>"
                    colorb = str(hex(int((1 - op.item()) * 0 + 250)))[2:].upper()
                    colorg = str(hex(int((1 - op.item()) * 200)))[2:].upper()
                    colorr = str(hex(int((1 - op.item()) * 200)))[2:].upper()
                    if len(colorb) == 1: colorb = '0' + colorb
                    if len(colorg) == 1: colorg = '0' + colorg
                    if len(colorr) == 1: colorr = '0' + colorr
                    toks_list[c][t] = f"""<span style="color:#{colorr}{colorg}{colorb}"><u>""" + toks_list[c][t] + "</u></span>"
        elif i == 3 or (i > 3 and (i % 3 == 0)):
            # tok
            next_toks_list = [tgt_dict.string(
                history_toks[i + 1]["tokens"], None, extra_symbols_to_ignore=None
            ).split(" ")]
            for t in range(len(toks_list[0])):
                if toks_list[0][t] == "<unk>":
                    next_token = next_toks_list[0][t] if t < len(next_toks_list[0]) else "????"
                    toks_list[0][t] = """<span style="color:#00EE00"><u>""" + next_token + "</u></span>"
        with open(filename, 'a') as f:
            name = "tok" if i == 3 or (i > 3 and (i % 3 == 0)) else history_ops[iii]['name']
            f.write("<hr>\n")
            f.write(f"<p><b>{name}</b></p>")
            f.write("<table>\n")
            for c in range(len(toks_list)):
                f.write("<tr>")
                for tok in toks_list[c]:
                    f.write(f"<td>{tok}</td>")
                f.write("</tr>\n")
            f.write("</table>\n")


def get_precision_score(hyp, tgt_tokens, origin, pad=1, eos=2, bos=0):
    origin = origin[:hyp.size(0)]
    # print("origin:\n", origin.shape, "\n", origin, file=sys.stderr)
    # print("hyp:\n", hyp.shape, "\n", hyp, file=sys.stderr, flush=True)
    prev_origin_msk = (origin > 0) & hyp.ne(pad) & hyp.ne(bos) & hyp.ne(eos)
    pred_origin_msk = (origin == 0) & hyp.ne(pad) & hyp.ne(bos) & hyp.ne(eos)
    prev_vals, prev_cpt = hyp[prev_origin_msk].unique(return_counts=True)
    pred_vals, pred_cpt = hyp[pred_origin_msk].unique(return_counts=True)
    tgt_vals, tgt_cpt = tgt_tokens.unique(return_counts=True)


    # tgt_size = (tgt_tokens.ne(pad) & tgt_tokens.ne(bos) & tgt_tokens.ne(eos)).sum()
    # assert tgt_size >= prev_origin_msk.sum() + pred_origin_msk.sum(), f"{tgt_size} > {prev_origin_msk.sum() + pred_origin_msk.sum()}"

    # print("prev_origin_msk", prev_origin_msk)
    # print("pred_origin_msk", pred_origin_msk)

    prev_dict = dict(zip(prev_vals.tolist(), prev_cpt.tolist()))
    pred_dict = dict(zip(pred_vals.tolist(), pred_cpt.tolist()))
    tgt_dict = dict(zip(tgt_vals.tolist(), tgt_cpt.tolist()))

    # print("prev_dict", prev_dict)
    # print("pred_dict", pred_dict)
    # print("tgt_dict", tgt_dict)

    prev_precision = 0.
    for tok in prev_dict:
        if tok in tgt_dict:
            prev_precision += min(prev_dict[tok], tgt_dict[tok])
    prev_precision /= prev_origin_msk.sum()

    pred_precision = 0.
    for tok in pred_dict:
        if tok in tgt_dict:
            pred_precision += min(pred_dict[tok], tgt_dict[tok])
    pred_precision /= pred_origin_msk.sum()

    return prev_precision, pred_precision, prev_origin_msk.sum(), pred_origin_msk.sum()


def get_bigram_precision_score(hyp, tgt_tokens, origin, pad=1, eos=2, bos=0):
    origin = origin[:hyp.size(0)]
    # print("origin:\n", origin.shape, "\n", origin, file=sys.stderr)
    # print("hyp:\n", hyp.shape, "\n", hyp, file=sys.stderr, flush=True)
    prev_origin_msk = (origin > 0) & hyp.ne(pad) & hyp.ne(bos) & hyp.ne(eos)
    pred_origin_msk = (origin == 0) & hyp.ne(pad) & hyp.ne(bos) & hyp.ne(eos)

    prev_prev_origin_msk = prev_origin_msk[:-1] & prev_origin_msk[1:]
    prev_pred_origin_msk = prev_origin_msk[:-1] & pred_origin_msk[1:]
    pred_prev_origin_msk = pred_origin_msk[:-1] & prev_origin_msk[1:]
    pred_pred_origin_msk = pred_origin_msk[:-1] & pred_origin_msk[1:]

    bigram_hyp = torch.vstack((hyp[:-1], hyp[1:])).t()
    bigram_tgt = torch.vstack((tgt_tokens[:-1], tgt_tokens[1:])).t()

    prev_prev_vals, prev_prev_cpt = bigram_hyp[prev_prev_origin_msk].unique(return_counts=True, dim=0)
    prev_pred_vals, prev_pred_cpt = bigram_hyp[prev_pred_origin_msk].unique(return_counts=True, dim=0)
    pred_prev_vals, pred_prev_cpt = bigram_hyp[pred_prev_origin_msk].unique(return_counts=True, dim=0)
    pred_pred_vals, pred_pred_cpt = bigram_hyp[pred_pred_origin_msk].unique(return_counts=True, dim=0)
    bigram_tgt_vals, bigram_tgt_cpt = bigram_tgt.unique(return_counts=True, dim=0)


    # tgt_size = (tgt_tokens.ne(pad) & tgt_tokens.ne(bos) & tgt_tokens.ne(eos)).sum()
    # assert tgt_size >= prev_origin_msk.sum() + pred_origin_msk.sum(), f"{tgt_size} > {prev_origin_msk.sum() + pred_origin_msk.sum()}"

    # print("prev_origin_msk", prev_origin_msk)
    # print("pred_origin_msk", pred_origin_msk)

    prev_prev_dict = dict(zip([tuple(e) for e in prev_prev_vals.tolist()], prev_prev_cpt.tolist()))
    prev_pred_dict = dict(zip([tuple(e) for e in prev_pred_vals.tolist()], prev_pred_cpt.tolist()))
    pred_prev_dict = dict(zip([tuple(e) for e in pred_prev_vals.tolist()], pred_prev_cpt.tolist()))
    pred_pred_dict = dict(zip([tuple(e) for e in pred_pred_vals.tolist()], pred_pred_cpt.tolist()))
    bigram_tgt_dict = dict(zip([tuple(e) for e in bigram_tgt_vals.tolist()], bigram_tgt_cpt.tolist()))

    # print("prev_dict", prev_dict)
    # print("pred_dict", pred_dict)
    # print("tgt_dict", tgt_dict)

    prev_prev_precision = 0.
    for tok in prev_prev_dict:
        if tok in bigram_tgt_dict:
            prev_prev_precision += min(prev_prev_dict[tok], bigram_tgt_dict[tok])
    prev_prev_precision /= prev_prev_origin_msk.sum()

    prev_pred_precision = 0.
    for tok in prev_pred_dict:
        if tok in bigram_tgt_dict:
            prev_pred_precision += min(prev_pred_dict[tok], bigram_tgt_dict[tok])
    prev_pred_precision /= prev_pred_origin_msk.sum()

    pred_prev_precision = 0.
    for tok in pred_prev_dict:
        if tok in bigram_tgt_dict:
            pred_prev_precision += min(pred_prev_dict[tok], bigram_tgt_dict[tok])
    pred_prev_precision /= pred_prev_origin_msk.sum()

    pred_pred_precision = 0.
    for tok in pred_pred_dict:
        if tok in bigram_tgt_dict:
            pred_pred_precision += min(pred_pred_dict[tok], bigram_tgt_dict[tok])
    pred_pred_precision /= pred_pred_origin_msk.sum()

    return prev_prev_precision, prev_pred_precision, pred_prev_precision, pred_pred_precision, prev_prev_origin_msk.sum(), prev_pred_origin_msk.sum(), pred_prev_origin_msk.sum(), pred_pred_origin_msk.sum()


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]


def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]


def convert_padding_direction(
    src_tokens, padding_idx, right_to_left: bool = False, left_to_right: bool = False
):
    assert right_to_left ^ left_to_right
    pad_mask = src_tokens.eq(padding_idx)
    if not pad_mask.any():
        # no padding, return early
        return src_tokens
    if left_to_right and not pad_mask[:, 0].any():
        # already right padded
        return src_tokens
    if right_to_left and not pad_mask[:, -1].any():
        # already left padded
        return src_tokens
    max_len = src_tokens.size(1)
    buffered = torch.empty(0).long()
    if max_len > 0:
        torch.arange(max_len, out=buffered)
    range = buffered.type_as(src_tokens).expand_as(src_tokens)
    num_pads = pad_mask.long().sum(dim=1, keepdim=True)
    if right_to_left:
        index = torch.remainder(range - num_pads, max_len)
    else:
        index = torch.remainder(range + num_pads, max_len)
    return src_tokens.gather(1, index)


def item(tensor):
    # tpu-comment: making this a no-op for xla devices.
    if torch.is_tensor(tensor) and tensor.device.type == 'xla':
        return tensor.detach()
    if hasattr(tensor, "item"):
        return tensor.item()
    if hasattr(tensor, "__getitem__"):
        return tensor[0]
    return tensor


def multi_tensor_total_norm(grads, chunk_size=2048 * 32) -> torch.Tensor:
    per_device_grads = {}
    norms = []
    for grad in grads:
        device = grad.device
        cur_device_grads = per_device_grads.get(device)
        if cur_device_grads is None:
            cur_device_grads = []
            per_device_grads[device] = cur_device_grads
        cur_device_grads.append(grad)
    for device in per_device_grads.keys():
        cur_device_grads = per_device_grads[device]
        if device.type == "cuda":
            # TODO(msb) return has_inf
            has_inf = torch.zeros((1, 1), dtype=torch.int, device=device)
            with torch.cuda.device(device):
                norm = multi_tensor_l2norm(
                    chunk_size, has_inf, [cur_device_grads], False
                )
            norms.append(norm[0].to(torch.cuda.current_device()))
        else:
            norms += [torch.norm(g, p=2, dtype=torch.float32) for g in cur_device_grads]
    total_norm = torch.norm(torch.stack(norms))
    return total_norm


@torch.no_grad()
def clip_grad_norm_(params, max_norm, aggregate_norm_fn=None) -> torch.Tensor:
    def grad_exists(p):
        return p is not None and getattr(p, "grad", None) is not None
    if isinstance(params, torch.Tensor):
        params = [params]
    params = list(params)
    grads = [p.grad.detach() for p in params if grad_exists(p) and not hasattr(p, 'expert')]
    expert_grads = [p.grad.detach() for p in params if grad_exists(p) and hasattr(p, 'expert')]

    if len(grads) == 0:
        if len(params) > 0:
            return params[0].new_tensor(0.0)
        else:
            return torch.tensor(0.0)

    if len(grads) == 1:
        total_norm = torch.norm(grads[0], p=2, dtype=torch.float32)
    else:
        if multi_tensor_l2norm_available:
            total_norm = multi_tensor_total_norm(grads)
        else:
            if torch.cuda.is_available():
                warnings.warn(
                    "amp_C fused kernels unavailable, disabling multi_tensor_l2norm; "
                    "you may get better performance by installing NVIDIA's apex library"
                )
                device = torch.cuda.current_device()
            elif grads[0].device.type == "xla":
                device = grads[0].device
            else:
                device = torch.device("cpu")
            total_norm = torch.norm(
                torch.stack(
                    [torch.norm(g, p=2, dtype=torch.float32).to(device) for g in grads]
                )
            )

    if aggregate_norm_fn is not None:
        total_norm = aggregate_norm_fn(total_norm)

    if max_norm > 0:
        max_norm = float(max_norm)
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp_(max=1)
        for g in grads + expert_grads:
            g.mul_(clip_coef)
    return total_norm


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _match_types(arg1, arg2):
    """Convert the numerical argument to the same type as the other argument"""

    def upgrade(arg_number, arg_structure):
        if isinstance(arg_structure, tuple):
            return tuple([arg_number] * len(arg_structure))
        elif isinstance(arg_structure, dict):
            arg = copy.deepcopy(arg_structure)
            for k in arg:
                arg[k] = upgrade(arg_number, arg_structure[k])
            return arg
        else:
            return arg_number

    if isinstance(arg1, float) or isinstance(arg1, int):
        return upgrade(arg1, arg2), arg2
    elif isinstance(arg2, float) or isinstance(arg2, int):
        return arg1, upgrade(arg2, arg1)

    return arg1, arg2


def resolve_max_positions(*args):
    """Resolve max position constraints from multiple sources."""

    def map_value_update(d1, d2):
        updated_value = copy.deepcopy(d1)
        for key in d2:
            if key not in updated_value:
                updated_value[key] = d2[key]
            else:
                updated_value[key] = min(d1[key], d2[key])
        return updated_value

    def nullsafe_min(l):
        minim = None
        for item in l:
            if minim is None:
                minim = item
            elif item is not None and item < minim:
                minim = item
        return minim

    max_positions = None
    for arg in args:
        if max_positions is None:
            max_positions = arg
        elif arg is not None:
            max_positions, arg = _match_types(max_positions, arg)
            if isinstance(arg, float) or isinstance(arg, int):
                max_positions = min(max_positions, arg)
            elif isinstance(arg, dict):
                max_positions = map_value_update(max_positions, arg)
            else:
                max_positions = tuple(map(nullsafe_min, zip(max_positions, arg)))

    return max_positions


def import_user_module(args):
    module_path = getattr(args, "user_dir", None)
    if module_path is not None:
        module_path = os.path.abspath(args.user_dir)
        if not os.path.exists(module_path) and not os.path.isfile(os.path.dirname(module_path)):
            fairseq_rel_path = os.path.join(os.path.dirname(__file__), args.user_dir)
            if os.path.exists(fairseq_rel_path):
                module_path = fairseq_rel_path
            else:
                fairseq_rel_path = os.path.join(
                    os.path.dirname(__file__), "..", args.user_dir
                )
                if os.path.exists(fairseq_rel_path):
                    module_path = fairseq_rel_path
                else:
                    raise FileNotFoundError(module_path)

        # ensure that user modules are only imported once
        import_user_module.memo = getattr(import_user_module, "memo", set())
        if module_path not in import_user_module.memo:
            import_user_module.memo.add(module_path)

            module_parent, module_name = os.path.split(module_path)
            if module_name not in sys.modules:
                sys.path.insert(0, module_parent)
                importlib.import_module(module_name)
            else:
                raise ImportError(
                    "Failed to import --user-dir={} because the corresponding module name "
                    "({}) is not globally unique. Please rename the directory to "
                    "something unique and try again.".format(module_path, module_name)
                )


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def log_softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.log_softmax(x.float(), dim=dim)
    else:
        return F.log_softmax(x, dim=dim, dtype=torch.float32)


def get_perplexity(loss, round=2, base=2):
    from fairseq.logging.meters import safe_round

    if loss is None:
        return 0.0
    try:
        return safe_round(base ** loss, round)
    except OverflowError:
        return float("inf")


def deprecation_warning(message, stacklevel=3):
    # don't use DeprecationWarning, since it's ignored by default
    warnings.warn(message, stacklevel=stacklevel)


def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_fast":
        deprecation_warning(
            "--activation-fn=gelu_fast has been renamed to gelu_accurate"
        )
        return gelu_accurate
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))


def get_available_activation_fns() -> List:
    return [
        "relu",
        "gelu",
        "gelu_fast",  # deprecated
        "gelu_accurate",
        "tanh",
        "linear",
    ]


@contextlib.contextmanager
def model_eval(model):
    is_training = model.training
    model.eval()
    yield
    model.train(is_training)


def has_parameters(module):
    try:
        next(module.parameters())
        return True
    except StopIteration:
        return False


def get_rng_state():
    state = {"torch_rng_state": torch.get_rng_state()}
    if xm is not None:
        state["xla_rng_state"] = xm.get_rng_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state()
    return state


def set_rng_state(state):
    torch.set_rng_state(state["torch_rng_state"])
    if xm is not None:
        xm.set_rng_state(state["xla_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda_rng_state"])


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = get_rng_state()

        torch.manual_seed(seed)
        if xm is not None:
            xm.set_rng_state(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        set_rng_state(self.rng_state)


def parse_alignment(line):
    """
    Parses a single line from the alingment file.

    Args:
        line (str): String containing the alignment of the format:
            <src_idx_1>-<tgt_idx_1> <src_idx_2>-<tgt_idx_2> ..
            <src_idx_m>-<tgt_idx_m>. All indices are 0 indexed.

    Returns:
        torch.IntTensor: packed alignments of shape (2 * m).
    """
    alignments = line.strip().split()
    parsed_alignment = torch.IntTensor(2 * len(alignments))
    for idx, alignment in enumerate(alignments):
        src_idx, tgt_idx = alignment.split("-")
        parsed_alignment[2 * idx] = int(src_idx)
        parsed_alignment[2 * idx + 1] = int(tgt_idx)
    return parsed_alignment


def get_token_to_word_mapping(tokens, exclude_list):
    n = len(tokens)
    word_start = [int(token not in exclude_list) for token in tokens]
    word_idx = list(accumulate(word_start))
    token_to_word = {i: word_idx[i] for i in range(n)}
    return token_to_word


def extract_hard_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (
        ((tgt_sent != pad) & (tgt_sent != eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_invalid = (
        ((src_sent == pad) | (src_sent == eos)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    src_token_to_word = get_token_to_word_mapping(src_sent, [eos, pad])
    tgt_token_to_word = get_token_to_word_mapping(tgt_sent, [eos, pad])
    alignment = []
    if len(tgt_valid) != 0 and len(src_invalid) < len(src_sent):
        attn_valid = attn[tgt_valid]
        attn_valid[:, src_invalid] = float("-inf")
        _, src_indices = attn_valid.max(dim=1)
        for tgt_idx, src_idx in zip(tgt_valid, src_indices):
            alignment.append(
                (
                    src_token_to_word[src_idx.item()] - 1,
                    tgt_token_to_word[tgt_idx.item()] - 1,
                )
            )
    return alignment


def extract_soft_alignment(attn, src_sent, tgt_sent, pad, eos):
    tgt_valid = (
        ((tgt_sent != pad)).nonzero(as_tuple=False)
    )
    src_valid = (
        ((src_sent != pad)).nonzero(as_tuple=False).squeeze(dim=-1)
    )
    alignment = []
    if len(tgt_valid) != 0 and len(src_valid) != 0:
        attn_valid = attn[tgt_valid, src_valid]
        alignment = [
            ["{:.6f}".format(p) for p in src_probs.tolist()]
            for src_probs in attn_valid
        ]
    return alignment


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def get_tpu_device():
    return xm.xla_device()


def tpu_data_loader(itr):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    from fairseq.data import iterators

    xm.rendezvous("tpu_data_loader")  # wait for all workers
    xm.mark_step()
    device = xm.xla_device()
    return iterators.CountingIterator(
        pl.ParallelLoader(itr, [device]).per_device_loader(device),
        start=getattr(itr, "n", 0),
        total=len(itr),
    )


def is_xla_tensor(tensor):
    return torch.is_tensor(tensor) and tensor.device.type == 'xla'


def index_put(tensor, indices, value):
    if is_xla_tensor(tensor):
        for _ in range(indices.dim(), tensor.dim()):
            indices = indices.unsqueeze(-1)
        if indices.size(-1) < tensor.size(-1):
            indices = indices.expand_as(tensor)
        tensor = torch.mul(tensor, ~indices) + torch.mul(value, indices)
    else:
        tensor[indices] = value
    return tensor


def xla_device_to_cpu(dat):
    import torch_xla.core.xla_model as xm
    return xm._maybe_convert_to_cpu(dat)


class CudaEnvironment(object):
    def __init__(self):
        cur_device = torch.cuda.current_device()
        prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
        self.name = prop.name
        self.major = prop.major
        self.minor = prop.minor
        self.total_memory_in_GB = prop.total_memory / 1024 / 1024 / 1024

    @staticmethod
    def pretty_print_cuda_env_list(cuda_env_list):
        """
        Given a list of CudaEnviorments, pretty print them
        """
        num_workers = len(cuda_env_list)
        center = "CUDA enviroments for all {} workers".format(num_workers)
        banner_len = 40 - len(center) // 2
        first_line = "*" * banner_len + center + "*" * banner_len
        logger.info(first_line)
        for r, env in enumerate(cuda_env_list):
            logger.info(
                "rank {:3d}: ".format(r)
                + "capabilities = {:2d}.{:<2d} ; ".format(env.major, env.minor)
                + "total memory = {:.3f} GB ; ".format(env.total_memory_in_GB)
                + "name = {:40s}".format(env.name)
            )
        logger.info(first_line)


def csv_str_list(x):
    return x.split(",")


def eval_str_list(x, type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(type, x))
    except TypeError:
        return [type(x)]


def eval_str_dict(x, type=dict):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x


def eval_bool(x, default=False):
    if x is None:
        return default
    try:
        return bool(eval(x))
    except TypeError:
        return default


class First(object):
    def __init__(self):
        super(First, self).__init__()
        print("first")

class Second(object):
    def __init__(self):
        super(Second, self).__init__()
        print("second")

class Third(First, Second):
    def __init__(self):
        super(Third, self).__init__()
        print("third")