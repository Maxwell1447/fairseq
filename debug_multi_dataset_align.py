import sys
import time
import torch

from fairseq.tasks.translation_multi_lev import load_lang_multi_dataset
from fairseq.data import Dictionary
from fairseq.models.nat import *
from fairseq.data import FairseqDataset, data_utils, iterators
from tqdm import tqdm
from fairseq import options, tasks, checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
from fairseq.data.multi_source_dataset import collate


# src_dict = Dictionary.load(
#     "/mnt/beegfs/home/bouthors/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-fr-en/dict.fr.txt"
# )
# tgt_dict = Dictionary.load(
#     "/mnt/beegfs/home/bouthors/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-fr-en/dict.en.txt"
# )

src_dict = Dictionary.load(
    "/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin/dict.fr.txt"
)
tgt_dict = Dictionary.load(
    "/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin/dict.en.txt"
)


def get_batch_iter(
    dataset,
    max_tokens=2048,
    ignore_invalid_inputs=True,
    required_batch_size_multiple=1,
    seed=1,
    num_shards=1,
    shard_id=0,
    num_workers=0,
    max_positions=100000,
    max_sentences=100000,
    epoch=1,
    data_buffer_size=0,
    disable_iterator_cache=False,
):
    assert isinstance(dataset, FairseqDataset)

    # initialize the dataset with the correct starting epoch
    dataset.set_epoch(epoch)

    # get indices ordered by example size
    with data_utils.numpy_seed(seed):
        indices = dataset.ordered_indices()

    # filter examples that are too large
    if max_positions is not None:
        indices, _ = dataset.filter_indices_by_size(
            indices, max_positions, max_acceptable_retrieved_ratio=10.0
        )

    # create mini-batches with given size constraints
    batch_sampler = dataset.batch_by_size(
        indices,
        max_tokens=max_tokens,
        max_sentences=max_sentences,
        required_batch_size_multiple=required_batch_size_multiple,
    )

    # return a reusable, sharded iterator
    epoch_iter = iterators.EpochBatchIterator(
        dataset=dataset,
        collate_fn=dataset.collater,
        batch_sampler=batch_sampler,
        seed=seed,
        num_shards=num_shards,
        shard_id=shard_id,
        num_workers=num_workers,
        epoch=epoch,
        buffer_size=data_buffer_size,
    )

    return epoch_iter


def regularize_shapes(ys, y):
    # print("yyys", ys)
    # print("y")
    bsz = y.size(0)
    M = max(ys.size(-1), y.size(-1))
    N = ys.size(1)
    shape = (bsz, N + 1, M)
    X = y.new(*shape).fill_(tgt_dict.pad())
    X[:, -1, : y.size(-1)] = y
    X[:, :-1, : ys.size(-1)] = ys

    return X[:, :-1, :], X[:, -1, :]


def regularize_shape_multi(ys):
    # ys: list of (L_i,)
    M = max([len(y) for y in ys])
    N = len(ys)
    shape = (N, M)
    Y = ys[0].new(*shape).fill_(tgt_dict.pad())
    for n in range(N):
        Y[n, :len(ys[n])] = ys[n]

    return Y



def load_model():
    from fairseq import models
    import argparse
    import ast
    parser = options.get_generation_parser()
    # print(parser)

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # parser.add_argument("data", default="/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-noised")
    parser.add_argument("--arch", default="multi_levenshtein_transformer")
    parser.add_argument("--task", default="multi_translation_lev")
    # parser.add_argument("--num-retrieved", default=1, type=int)
    parser.add_argument("--criterion", default="nat_loss")
    parser.add_argument("--ddp-backend", default="legacy_ddp")
    parser.add_argument("--batch-size", default=1, type=int)
    # parser.add_argument("--max-valency", default=5, type=int)
    # parser.add_argument("--share-all-embedding", action="store_true")
    # parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument(
        "--path", 
        default="/gpfswork/rech/usb/ufn16wp/NLP4NLP/scripts/multi-lev/models-small/transformer-multi-lev-fr-en-ecb-3NN/checkpoint_last.pt"
    )
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    # print(args)
    # print(cfg)
    # print(cfg.keys())
    task = tasks.setup_task(cfg.task)
    # model = task.build_model(cfg.model)

    overrides = ast.literal_eval(cfg.common_eval.model_overrides)
    models, saved_cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        arg_overrides=overrides,
        task=task,
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
    )

    # criterion = LabelSmoothedDualImitationCriterion(task, 0.1)

    # print(overrides)

    # model = models.build_model(cfg.model, task)
    # print(len(models))
    return models[0] #, criterion


def compute_loss(
    outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
):
    """
    outputs: batch x len x d_model
    targets: batch x len
    masks:   batch x len

    policy_logprob: if there is some policy
        depends on the likelihood score as rewards.
    """

    def mean_ds(x, dim=None):
        return (
            x.float().mean().type_as(x)
            if dim is None
            else x.float().mean(dim).type_as(x)
        )

    if masks is not None:
        print("+++++++++++", name)
        # print("out", outputs.shape, outputs.device, outputs.dtype)
        # print("tgt", targets.shape, targets.device, targets.dtype)
        # print("masks", masks.shape, masks.device, masks.dtype)
        # print()

        outputs = outputs[masks]
        targets = targets[masks]

    if masks is not None and not masks.any():
        nll_loss = outputs.new_tensor(0)
        # nll_loss = torch.tensor(0, device=outputs.device, dtype=outputs.dtype)
        loss = nll_loss
    else:
        logits = F.log_softmax(outputs, dim=-1)
        if targets.dim() == 1:
            # print("logits", logits.shape, logits.dtype)
            # print("logits", "min", logits.min(), "max", logits.max())
            # print("targets", targets.shape, targets.dtype)
            # print("targets", "min", targets.min(), "max", targets.max())
            if name == "del-loss":
                select_zero = targets == 0
                print("pred  ", logits.argmax(-1)[select_zero][:20])
                print("target", targets[select_zero][:20])
            elif name == "plh-loss":
                select_non_zero = targets.ne(0)
                print("pred  ", logits.argmax(-1)[select_non_zero][:20])
                print("target", targets[select_non_zero][:20])
            else:
                print("pred  ", logits.argmax(-1)[:20])
                print("target", targets[:20])
            losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

        else:  # soft-labels
            losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
            while losses.dim() > 1:
                losses = losses.sum(-1)

        nll_loss = mean_ds(losses)
        if label_smoothing > 0:
            loss = (
                nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
            )
        else:
            loss = nll_loss

    loss = loss * factor
    return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

def forward_loss(outputs):
    """Compute the loss for the given sample.
    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """
    losses, nll_loss = [], []

    for obj in outputs:
        # print(obj)
        _losses = compute_loss(
            outputs[obj].get("out"),
            outputs[obj].get("tgt"),
            outputs[obj].get("mask", None),
            outputs[obj].get("ls", 0.0),
            name=obj + "-loss",
            factor=outputs[obj].get("factor", 1.0),
        )

        losses += [_losses]
        if outputs[obj].get("nll_loss", False):
            nll_loss += [_losses.get("nll_loss", 0.0)]

    loss = sum(l["loss"] for l in losses)
    nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

    return loss

def test_artificial_align(sample_=None, k=1, max_valency=1):

    def get_mask_from_prob(bsz, p):
        return torch.rand(bsz) > p

    def combine_res(res1, res2, mask):
        res = dict()
        for key in res1:
            shape = [i for i in res1[key].shape]
            shape[0] += res2[key].size(0)
            res[key] = res1[key].new_empty(shape)
            res[key][mask] = res1[key]
            res[key][~mask] = res2[key]
        return res
    
    if sample_ is None:
        sample = dict()
        sample["multi_source"] = torch.tensor([[0, 7, 9, 6, 4, 2, 1], [0, 9, 7, 6, 4, 9, 2]], dtype=torch.int64).unsqueeze(0)
        sample["target"] = torch.tensor([[0, 7, 4, 5, 6, 2, 1]], dtype=torch.int64)
        sample["multi_source"] = torch.randint(5, 15, size=(2, 40), dtype=torch.int64).unsqueeze(0)
        sample["target"] = torch.randint(5, 15, size=(1, 40), dtype=torch.int64)
    else:
        sample = sample_
        sample["multi_source"] = regularize_shape_multi(sample["multi_source"]).unsqueeze(0)
        sample["target"] = sample["target"].unsqueeze(0)
        sample["multi_source"], sample["target"] = regularize_shapes(sample["multi_source"], sample["target"])
        # print(sample)
        # print(sample["multi_source"].shape)
        # print("(T)  >>>", tgt_dict.string(sample_extreme["target"]))
        # print("(S)  >>>", tgt_dict.string(sample_extreme["source"]))
        # print("(S1) >>>", tgt_dict.string(sample_extreme["multi_source"][0]))
    # print(sample["multi_source"].shape, sample["target"].shape)
    # print(sample["multi_source"].shape)
    # print(sample["target"].shape)
    y_init_star, tgt_tokens = sample["multi_source"], sample["target"] 
    
    mask_star = get_mask_from_prob(y_init_star.size(0), 0.2 * 0)
    t1 = time.time()
    res_star = pi_star(
        sample["multi_source"][mask_star],
        sample["target"][mask_star],
        k=k,
        max_valency=max_valency,
        pad_symbol=tgt_dict.pad(),
        plh_symbol=tgt_dict.unk(),
        Kmax=64,
        device="cpu",
    )
    t2 = time.time()
    # mask_debug = ~((res_star["y_cmb"] == 2).sum(-1).sum(-1) == res_star["y_cmb"].size(1))
    # print(mask_debug)
    print(res_star["del_tgt"])
    print(res_star["y_plh"])
    cov = (1 - (res_star["y_tok"] == tgt_dict.unk()).sum() / res_star["y_tok"].ne(tgt_dict.pad()).sum()).item()

    return (t2 - t1), cov
    print("execution time = ", t2 - t1)
    res_del = pi_del(
        y_init_star[~mask_star].shape,
        tgt_tokens[~mask_star],
        pad_symbol=3,
        plh_symbol=1,
        bos_symbol=0,
        eos_symbol=2,
        Kmax=64,
        device=tgt_tokens.device,
    )
    res = combine_res(res_star, res_del, mask_star)

    print(res)

    y_plh = res["y_plh"]
    y_cmb = res["y_cmb"]
    y_tok = res["y_tok"]

    del_tgt = res["del_tgt"]
    del_mask = res["del_mask"]

    plh_tgt = res["plh_tgt"]
    plh_mask = res["plh_mask"]
    

    cmb_tgt = res["cmb_tgt"]
    cmb_mask = res["cmb_mask"]

    tok_tgt = res["tok_tgt"]
    tok_mask = res["tok_mask"]

    # y_cmb = pi_sel(
    #     y_cmb,
    #     y_init_star,
    #     0.2,
    #     pad_symbol=1,
    #     plh_symbol=3,
    #     bos_symbol=0,
    #     eos_symbol=2,
    #     device=tok_mask.device
    # )
    # y_cmb[0, 0, 1] = 11
    # y_cmb[0, 1, 4] = 11

    # cmb_tgt = handle_all_plh_case(cmb_tgt, y_tok, 3)
    msk_cmb_sel = ((y_tok == 3) & (~(y_cmb == 3).all(1))).unsqueeze(1).expand_as(cmb_tgt) & (y_cmb == 3)
    cmb_tgt[msk_cmb_sel] = 7
    print("y_cmb", y_cmb)
    print("cmb_tgt", cmb_tgt)

    mask_mask = get_mask_from_prob(y_tok.size(0), 0.2)   

    y_tok[~mask_mask], tok_tgt[~mask_mask], tok_mask[~mask_mask] = pi_mask(
        tok_tgt[~mask_mask],
        pad_symbol=1,
        plh_symbol=3,
        bos_symbol=0,
        eos_symbol=2,
        device=tok_mask.device
    )

    del_out = torch.rand(1, sample["multi_source"].size(1), sample["multi_source"].size(2), 2)
    plh_out = torch.rand(1, sample["multi_source"].size(1), sample["multi_source"].size(2) - 1, 64)
    cmb_out = torch.rand(1, sample["multi_source"].size(1), sample["multi_source"].size(2), 2)
    tok_out = torch.rand(1, sample["multi_source"].size(2), 35000)

    # print(plh_tgt.shape, plh_mask.shape, plh_out.shape)

    for_loss = {
        "plh": {"out": plh_out, "tgt": plh_tgt, "mask": plh_mask, "ls": 0.01,},
        "tok": {
            "out": tok_out,
            "tgt": tok_tgt,
            "mask": tok_mask,
            "ls": 0.1,
            "nll_loss": True,
        },
        "del": {"out": del_out, "tgt": del_tgt, "mask": del_mask,},
        "cmb": {"out": cmb_out, "tgt": cmb_tgt, "mask": cmb_mask,},
    }

    # forward_loss(for_loss)

lmd = load_lang_multi_dataset(
    "/gpfswork/rech/usb/ufn16wp/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin",
    "train",
    "fr",
    src_dict,
    "en",
    tgt_dict,
    3,
    True,
    "mmap",
    -1,
    True,
    False,
    1024,
    1024,
    prepend_bos=True,
)

# 139552, 154425
# print(lmd[162264])
# dt, cov = test_artificial_align(sample_=lmd[154425], max_valency=1, k=1)

# for max_valency in [1, 5, 10, -1]:
#     for k in [1]:
#         # test_artificial_align()
#         dts = list()
#         covs = list()
#         # k = 10
#         # max_valency = -1
#         print("k =", k, "; max_valency =", max_valency)
#         for i in tqdm(range(min(len(lmd), 6500))):
#             dt, cov = test_artificial_align(lmd[i], max_valency=max_valency, k=k)
#             dts.append(dt)
#             covs.append(cov)

#         dts = np.array(dts)
#         covs = np.array(covs)
#         print("total time:", dts.sum())
#         print("mean cov:  ", covs.mean())
#         print()

# for i in range(78835):
#     if (lmd[i]["target"] == 3).any():
#         print(i, tgt_dict.string(lmd[i]["target"]))

# sample_extreme = lmd[78835]
# sample_extreme = lmd[12]

# print(sample_extreme)
# print("(T)  >>>", tgt_dict.string(sample_extreme["target"]))
# print("(S)  >>>", tgt_dict.string(sample_extreme["source"]))
# print("(S1) >>>", tgt_dict.string(sample_extreme["multi_source"][0]))

device = "cuda"
device = "cpu"
model = load_model()
model.max_valency = 2
model = model.to(device)
# iterator_3000 = get_batch_iter(lmd)
# data_iter = iterator_3000.next_epoch_itr(shuffle=False)
# print(len(data_iter))
# for i, sample in tqdm(enumerate(data_iter)):

#     if i == 13074:
#         # continue

    
#         print(str(i))
#         print(sample["id"])
#         print(sample)

#         # B x L
#         src_tokens, src_lengths = (
#             sample["net_input"]["src_tokens"].to(device),
#             sample["net_input"]["src_lengths"].to(device),
#         )
#         sample["num_iter"] = i
#         tgt_tokens = sample["target"].to(device)
#         multi_src_tokens = sample["net_input"]["multi_src_tokens"].to(device)
#         outputs = model(src_tokens, src_lengths, multi_src_tokens, tgt_tokens, i, ids=sample["id"])
    
    # x = sample["net_input"]["src_tokens"]
    # tgt_tokens = sample["target"]
    # y_init_star = sample["net_input"]["multi_src_tokens"]
    # # outputs = model(src_tokens, multi_src_tokens, tgt_tokens)

    # y_init_star, tgt_tokens = regularize_shapes(
    #     y_init_star, tgt_tokens
    # )

    # # print("batch", i, "  with", x.size(0), "elements")

    # if i >= 0:
    #     # print(sample["id"].cpu().numpy())

    #     res = pi_star(
    #         y_init_star,
    #         tgt_tokens,
    #         pad_symbol=tgt_dict.pad(),
    #         plh_symbol=tgt_dict.unk(),
    #         Kmax=50,
    #         max_valency=10,
    #         device=x.device,
    #     )
    #     # break

model.full_mlevt = True
# sample = [lmd[154425], lmd[139552]] #, 154425 139552
sample = [lmd[154425]]
# sample = [lmd[139552]]
print(sample)
sample = collate(
    sample,
    tgt_dict.pad(),
    tgt_dict.eos(),
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
)
print(sample.keys())
src_tokens, src_lengths = (
    sample["net_input"]["src_tokens"].to(device),
    sample["net_input"]["src_lengths"].to(device),
)
sample["num_iter"] = 2
tgt_tokens = sample["target"].to(device)
multi_src_tokens = sample["net_input"]["multi_src_tokens"].to(device)
# outputs = model(src_tokens, src_lengths, multi_src_tokens, tgt_tokens, sample["num_iter"], ids=sample["id"])

multi_src_tokens, tgt_tokens = regularize_shapes(
    multi_src_tokens, tgt_tokens
)
res = pi_star(
    multi_src_tokens,
    tgt_tokens,
    pad_symbol=tgt_dict.pad(),
    plh_symbol=tgt_dict.unk(),
    Kmax=50,
    max_valency=10,
    device=src_tokens.device,
)

print(res)


# batch = next(data_iter)

# print(batch)


# # source, multi_source (list), target
# for i in range(50):
#     i = 1
#     src = lmd[i]["source"]
#     tgt = lmd[i]["target"]
#     multi_src = lmd[i]["multi_source"]
#     # print(i, (tgt[: len(multi_src[0])] == multi_src[0][: len(tgt)]).float().mean())

#     # print(lmd[i])
#     # print("src")
#     # print(src_dict.string(src, None, extra_symbols_to_ignore=None))
#     # print()
#     # print("tgt")
#     # print(tgt_dict.string(tgt, None, extra_symbols_to_ignore=None))
#     # print()
#     # print("multi src")
#     # for toks in multi_src:
#     #     print(tgt_dict.string(toks, None, extra_symbols_to_ignore=None))
#     #     print()
#     break

# multi_src = (
#     data_utils.collate_tokens_list(
#         [ssrc.unsqueeze(0) for ssrc in multi_src],
#         tgt_dict.pad(),
#         tgt_dict.eos(),
#         False,
#         False,
#         pad_to_length=None,
#         pad_to_multiple=1,
#     )
#     .squeeze()
#     .unsqueeze(0)
# )
# # print(tgt.shape)
# # print("###")
# # # tgt_ = torch.full((len(tgt) + 1,), tgt_dict.pad(), dtype=tgt.dtype, device=tgt.device)
# # # tgt_[:-1] = tgt
# # # tgt = tgt_
# y_init_star, tgt_tokens = regularize_shapes(
#     multi_src, tgt.unsqueeze(0)
# )

# # print("bos", tgt_dict.bos())
# # print("eos", tgt_dict.eos())
# # print("pad", tgt_dict.pad())
# # print("unk", tgt_dict.unk())

# # print("src_tokens")
# # print(x)

# # print("y_init_star")
# # print(y_init_star)

# # print("tgt_tokens")
# # print(tgt_tokens)

# res = pi_star(
#     y_init_star,
#     tgt_tokens,
#     pad_symbol=tgt_dict.pad(),
#     plh_symbol=tgt_dict.unk(),
#     Kmax=50,
#     device=x.device,
# )
# print("----------------- pi star ----------------")
# print(res)

# # res_del = pi_del(
# #     y_init_star.shape,
# #     tgt_tokens,
# #     pad_symbol=tgt_dict.pad(),
# #     plh_symbol=tgt_dict.unk(),
# #     bos_symbol=tgt_dict.bos(),
# #     eos_symbol=tgt_dict.eos(),
# #     Kmax=50,
# #     device=x.device,
# # )
# # print("----------------- pi del ----------------")
# # print(res_del)

# y_cmb = res["y_cmb"]

# res2 = pi_sel(
#     y_cmb,
#     y_init_star,
#     gamma=0.1,
#     pad_symbol=tgt_dict.pad(),
#     plh_symbol=tgt_dict.unk(),
#     bos_symbol=tgt_dict.bos(),
#     eos_symbol=tgt_dict.eos(),
#     device=x.device,
# )
# print("----------------- pi sel ----------------")
# print(res2)

# # print("-----------------")
# # print()

# # print("~" * 20 + "apply del" + "~" * 20)
# # print(res["y_plh"])
# # print(
# #     apply_del(
# #         y_init_star, res["del_tgt"], tgt_dict.pad(), tgt_dict.bos(), tgt_dict.eos()
# #     )
# # )

# # print("~" * 20 + "apply plh" + "~" * 20)
# # print(res["y_cmb"])
# # print(
# #     apply_plh(
# #         res["y_plh"], res["plh_tgt"], tgt_dict.pad(), tgt_dict.unk(), tgt_dict.eos()
# #     )
# # )

# # print("~" * 20 + "apply cmb" + "~" * 20)
# # print(res["y_tok"])
# # print(
# #     apply_cmb(
# #         res["y_cmb"],
# #         res["cmb_tgt"],
# #         tgt_dict.pad(),
# #         tgt_dict.bos(),
# #         tgt_dict.eos(),
# #         tgt_dict.unk(),
# #     )
# # )

# # print("~" * 20 + "apply cmb random" + "~" * 20)

# batch = next(data_iter)

# print(batch)


# # source, multi_source (list), target
# for i in range(50):
#     i = 1
#     src = lmd[i]["source"]
#     tgt = lmd[i]["target"]
#     multi_src = lmd[i]["multi_source"]
#     # print(i, (tgt[: len(multi_src[0])] == multi_src[0][: len(tgt)]).float().mean())

#     # print(lmd[i])
#     # print("src")
#     # print(src_dict.string(src, None, extra_symbols_to_ignore=None))
#     # print()
#     # print("tgt")
#     # print(tgt_dict.string(tgt, None, extra_symbols_to_ignore=None))
#     # print()
#     # print("multi src")
#     # for toks in multi_src:
#     #     print(tgt_dict.string(toks, None, extra_symbols_to_ignore=None))
#     #     print()
#     break

# multi_src = (
#     data_utils.collate_tokens_list(
#         [ssrc.unsqueeze(0) for ssrc in multi_src],
#         tgt_dict.pad(),
#         tgt_dict.eos(),
#         False,
#         False,
#         pad_to_length=None,
#         pad_to_multiple=1,
#     )
#     .squeeze()
#     .unsqueeze(0)
# )
# # print(tgt.shape)
# # print("###")
# # # tgt_ = torch.full((len(tgt) + 1,), tgt_dict.pad(), dtype=tgt.dtype, device=tgt.device)
# # # tgt_[:-1] = tgt
# # # tgt = tgt_
# x, y_init_star, tgt_tokens = regularize_shapes(
#     src.unsqueeze(0), multi_src, tgt.unsqueeze(0)
# )

# # print("bos", tgt_dict.bos())
# # print("eos", tgt_dict.eos())
# # print("pad", tgt_dict.pad())
# # print("unk", tgt_dict.unk())

# # print("src_tokens")
# # print(x)

# # print("y_init_star")
# # print(y_init_star)

# # print("tgt_tokens")
# # print(tgt_tokens)

# res = pi_star(
#     y_init_star,
#     tgt_tokens,
#     pad_symbol=tgt_dict.pad(),
#     plh_symbol=tgt_dict.unk(),
#     Kmax=50,
#     device=x.device,
# )
# print("----------------- pi star ----------------")
# print(res)

# # res_del = pi_del(
# #     y_init_star.shape,
# #     tgt_tokens,
# #     pad_symbol=tgt_dict.pad(),
# #     plh_symbol=tgt_dict.unk(),
# #     bos_symbol=tgt_dict.bos(),
# #     eos_symbol=tgt_dict.eos(),
# #     Kmax=50,
# #     device=x.device,
# # )
# # print("----------------- pi del ----------------")
# # print(res_del)

# y_cmb = res["y_cmb"]

# res2 = pi_sel(
#     y_cmb,
#     y_init_star,
#     gamma=0.1,
#     pad_symbol=tgt_dict.pad(),
#     plh_symbol=tgt_dict.unk(),
#     bos_symbol=tgt_dict.bos(),
#     eos_symbol=tgt_dict.eos(),
#     device=x.device,
# )
# print("----------------- pi sel ----------------")
# print(res2)

# # print("-----------------")
# # print()

# # print("~" * 20 + "apply del" + "~" * 20)
# # print(res["y_plh"])
# # print(
# #     apply_del(
# #         y_init_star, res["del_tgt"], tgt_dict.pad(), tgt_dict.bos(), tgt_dict.eos()
# #     )
# # )

# # print("~" * 20 + "apply plh" + "~" * 20)
# # print(res["y_cmb"])
# # print(
# #     apply_plh(
# #         res["y_plh"], res["plh_tgt"], tgt_dict.pad(), tgt_dict.unk(), tgt_dict.eos()
# #     )
# # )

# # print("~" * 20 + "apply cmb" + "~" * 20)
# # print(res["y_tok"])
# # print(
# #     apply_cmb(
# # print(
# #     apply_cmb(
# #         res["y_cmb"],
# #         cmb_tgt_rd,
# #         tgt_dict.pad(),
# #         tgt_dict.bos(),
# #         tgt_dict.eos(),
# #         tgt_dict.unk(),
# #     )
# )

