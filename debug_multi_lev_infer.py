import sys
from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
import torch

from fairseq.tasks.translation_multi_lev import load_lang_multi_dataset
from fairseq.data import Dictionary
from fairseq.models.nat import *
from fairseq.data import FairseqDataset, data_utils, iterators
from fairseq import options, tasks, checkpoint_utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.dataclass.configs import FairseqDataclass, GenerationConfig
from fairseq.criterions.nat_loss import LabelSmoothedDualImitationCriterion
from tqdm import tqdm


src_dict = Dictionary.load(
    "/mnt/beegfs/home/bouthors/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-fr-en/dict.fr.txt"
)
tgt_dict = Dictionary.load(
    "/mnt/beegfs/home/bouthors/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-fr-en/dict.en.txt"
)


def get_batch_iter(
    dataset,
    max_tokens=4096,
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
            indices, max_positions, max_acceptable_retrieved_ratio=1.2
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


def regularize_shapes(x, ys, y):
    # print("yyys", ys)
    # print("y")
    bsz = x.size(0)
    M = max(x.size(-1), ys.size(-1), y.size(-1))
    N = ys.size(1)
    shape = (bsz, N + 2, M)
    X = x.new(*shape).fill_(tgt_dict.pad())
    X[:, 0, : x.size(-1)] = x
    X[:, -1, : y.size(-1)] = y
    X[:, 1:-1, : ys.size(-1)] = ys

    return X[:, 0, :], X[:, 1:-1, :], X[:, -1, :]


def load_model():
    from fairseq import models
    import argparse
    import ast
    parser = options.get_generation_parser()
    # print(parser)

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    # parser.add_argument("data", default="/mnt/beegfs/projects/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin")
    parser.add_argument("--arch", default="multi_levenshtein_transformer")
    parser.add_argument("--task", default="multi_translation_lev")
    parser.add_argument("--num-retrieved", default=1, type=int)
    parser.add_argument("--criterion", default="nat_loss")
    parser.add_argument("--ddp-backend", default="legacy_ddp")
    parser.add_argument("--batch-size", default=1, type=int)
    # parser.add_argument("--share-all-embedding", action="store_true")
    # parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument(
        "--path", 
        default="/mnt/beegfs/projects/NLP4NLP/scripts/multi-lev/models-small/transformer-multi-lev-fr-en-debug/checkpoint_last.pt"
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

    criterion = LabelSmoothedDualImitationCriterion(task, 0.1)

    # print(overrides)

    # model = models.build_model(cfg.model, task)
    # print(len(models))
    return models[0], criterion


if __name__ == "__main__":

    lmd = load_lang_multi_dataset(
        "/mnt/beegfs/home/bouthors/NLP4NLP/DATA/multi-lev-DATA/ECB/data-bin-fr-en",
        "test",
        "fr",
        src_dict,
        "en",
        tgt_dict,
        1,
        True,
        "mmap",
        -1,
        True,
        False,
        1024,
        1024,
        prepend_bos=True,
    )

    model, criterion = load_model()
    model.eval()

    ir = IterativeRefinementGenerator(
        tgt_dict,
        models=[model],
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
    )

    




    # print(model)

    # for i in range(20):

    #     sample_extreme = lmd[i]

    #     # print(sample_extreme)
    #     print(sample_extreme["id"])
        
    #     print(tgt_dict.string(sample_extreme["source"]))

    #     print(tgt_dict.string(sample_extreme["multi_source"][0]))

    #     print(tgt_dict.string(sample_extreme["target"]))
    # for i in [4]:
    # for i in [19]:
    #     sample_extreme = lmd[i]

    #     print(sample_extreme)
        
    #     print(tgt_dict.string(sample_extreme["source"]))

    #     print(tgt_dict.string(sample_extreme["multi_source"][0]))

    #     print(tgt_dict.string(sample_extreme["target"]))

    #     x, ys, y = sample_extreme["source"], sample_extreme["multi_source"][0], sample_extreme["target"]
    #     M = max(x.size(-1), ys.size(-1), y.size(-1))
    #     shape = (3, M)
    #     X = x.new(*shape).fill_(tgt_dict.pad())
    #     X[0, : x.size(-1)] = x
    #     X[2, : y.size(-1)] = y
    #     X[1, : ys.size(-1)] = ys

    #     x, ys, y = X[0], X[1:-1], X[-1]


    #     res_star = pi_star(
    #         ys.unsqueeze(0),
    #         y.unsqueeze(0),
    #         pad_symbol=tgt_dict.pad(),
    #         plh_symbol=tgt_dict.unk(),
    #         Kmax=64,
    #         device="cpu",
    #     )

    #     src_lengths = torch.LongTensor(
    #         [sample_extreme["source"].ne(tgt_dict.pad()).long().sum()]
    #     )
    #     # print("src_lengths", src_lengths)
    #     # print("source", sample_extreme["source"])
    #     # print("pad", tgt_dict.pad())
    #     print(res_star["del_tgt"])
    #     print(res_star["del_mask"])
    #     # del_out = model.forward_debug(x.unsqueeze(0), ys.unsqueeze(0), src_lengths=src_lengths)["del_out"]
        
    #     with torch.no_grad():
    #         encoder_out = model.encoder(x.unsqueeze(0), src_lengths=src_lengths)
    #         del_out, _ = model.decoder.forward_del(
    #             normalize=True,
    #             prev_output_tokens=ys.unsqueeze(0),
    #             encoder_out=encoder_out,
    #         )
    #         plh_out, _ = model.decoder.forward_plh(
    #             normalize=True,
    #             prev_output_tokens=res_star["y_plh"],
    #             encoder_out=encoder_out,
    #         )
    #         cmb_out, _ = model.decoder.forward_cmb(
    #             normalize=True,
    #             prev_output_tokens=res_star["y_cmb"],
    #             encoder_out=encoder_out,
    #         )
    #         tok_out, _ = model.decoder.forward_tok(
    #             normalize=True,
    #             prev_output_tokens=res_star["y_tok"],
    #             encoder_out=encoder_out,
    #         )

    #         del_out_hard = del_out.argmax(-1)
    #         print(del_out.exp())
    #         print("del out", del_out_hard)
    #         print("del tgt", res_star["del_tgt"])
    #         print("plh out", plh_out.argmax(-1))
    #         print("plh tgt", res_star["plh_tgt"])
    #         print("cmb out", cmb_out.argmax(-1).squeeze(-1))
    #         print("cmb tgt", res_star["cmb_tgt"][0])
    #         print("tok out", tok_out.argmax(-1))
    #         print("tok tgt", res_star["tok_tgt"])
    #         # print(tok_out.argmax(-1)[res_star["tok_mask"]])
    #         # print(res_star["tok_mask"])
    #         # print(torch.argsort(tok_out[res_star["tok_mask"]], descending=True, dim=-1).squeeze()[:10])
    #         print(tgt_dict.string(torch.argsort(
    #             tok_out[res_star["tok_mask"]], 
    #             descending=True, 
    #             dim=-1
    #         ).squeeze()[:5]))
    #         print(tgt_dict.string(res_star["tok_tgt"][res_star["tok_mask"]]))
    #         print(sample_extreme.keys())

                
    device = torch.cuda.is_available()        

    iterator_3000 = get_batch_iter(lmd)
    data_iter = iterator_3000.next_epoch_itr(shuffle=False)

    print(len(data_iter))

    for _ in ir.generate_batched_itr(
            data_iter,
            maxlen_a=None,
            maxlen_b=None,
            cuda=False,
            timer=None,
            prefix_size=0,
        ):
        pass

    print("over")

    sys.exit(8)

    sample = next(data_iter)
    # print(sample.keys())
    # print(sample["net_input"].keys())
    # print(len(data_iter))
    cpt_del = 0
    cpt_plh = 0
    cpt_cmb = 0
    cpt_tok = 0
    cpt_del_zeros = 0
    cpt_plh_non_zeros = 0
    tot_del = 0
    tot_plh = 0
    tot_cmb = 0
    tot_tok = 0
    tot_del_zeros = 0
    tot_plh_non_zeros = 0
    print("unk = plh =", tgt_dict.unk())
    print("pad =", tgt_dict.pad())
    for i, sample in enumerate(data_iter):

        print(str(i), end="\r")
        if i > 4:
            break

        
        model.forward_decoder(
            decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
        )
        
        # x = sample["net_input"]["src_tokens"]
        # tgt_tokens = sample["target"]
        # y_init_star = sample["net_input"]["multi_src_tokens"]
        # # outputs = model(src_tokens, multi_src_tokens, tgt_tokens)

        # x, y_init_star, tgt_tokens = regularize_shapes(
        #     x, y_init_star, tgt_tokens
        # )

        # src_lengths = sample["net_input"]["src_lengths"]

        # with torch.no_grad():
        #     sample["prev_target"] = None
        #     criterion(model, sample)

        # with torch.no_grad():
        #     out = model(x, src_lengths, y_init_star, tgt_tokens)
        #     cpt_del += (out["del"]["out"][out["del"]["mask"]].argmax(-1) == out["del"]["tgt"][out["del"]["mask"]]).sum().item()
        #     cpt_plh += (out["plh"]["out"][out["plh"]["mask"]].argmax(-1) == out["plh"]["tgt"][out["plh"]["mask"]]).sum().item()
        #     cpt_cmb += (out["cmb"]["out"][out["cmb"]["mask"]].argmax(-1) == out["cmb"]["mask"][out["cmb"]["mask"]]).sum().item()
        #     cpt_tok += (out["tok"]["out"][out["tok"]["mask"]].argmax(-1) == out["tok"]["tgt"][out["tok"]["mask"]]).sum().item()
        #     cpt_del_zeros += (
        #         out["del"]["out"][out["del"]["mask"] & (out["del"]["tgt"] == 0)].argmax(-1) 
        #         == out["del"]["tgt"][out["del"]["mask"] & (out["del"]["tgt"] == 0)]
        #     ).sum().item()
        #     cpt_plh_non_zeros += (
        #         out["plh"]["out"][out["plh"]["mask"] & (out["plh"]["tgt"].ne(0))].argmax(-1) 
        #         == out["plh"]["tgt"][out["plh"]["mask"] & (out["plh"]["tgt"].ne(0))]
        #     ).sum().item()

        #     tot_del += out["del"]["mask"].sum().item()
        #     tot_plh += out["plh"]["mask"].sum().item()
        #     tot_cmb += out["cmb"]["mask"].sum().item()
        #     tot_tok += out["tok"]["mask"].sum().item()
        #     tot_del_zeros += (out["del"]["mask"] & (out["del"]["tgt"] == 0)).sum().item()
        #     tot_plh_non_zeros += (out["plh"]["mask"] & (out["plh"]["tgt"].ne(0))).sum().item()

        # res_star = pi_star(
        #     y_init_star,
        #     tgt_tokens,
        #     pad_symbol=tgt_dict.pad(),
        #     plh_symbol=tgt_dict.unk(),
        #     Kmax=64,
        #     device="cpu",
        # )

        # with torch.no_grad():
        #     encoder_out = model.encoder(x, src_lengths=src_lengths)
        #     del_out, _ = model.decoder.forward_del(
        #         normalize=True,
        #         prev_output_tokens=y_init_star,
        #         encoder_out=encoder_out,
        #     )
        #     plh_out, _ = model.decoder.forward_plh(
        #         normalize=True,
        #         prev_output_tokens=res_star["y_plh"],
        #         encoder_out=encoder_out,
        #     )
        #     cmb_out, _ = model.decoder.forward_cmb(
        #         normalize=True,
        #         prev_output_tokens=res_star["y_cmb"],
        #         encoder_out=encoder_out,
        #     )
        #     tok_out, _ = model.decoder.forward_tok(
        #         normalize=True,
        #         prev_output_tokens=res_star["y_tok"],
        #         encoder_out=encoder_out,
        #     )
        # cpt_del += (del_out[res_star["del_mask"]].argmax(-1) == res_star["del_tgt"][res_star["del_mask"]]).sum().item()
        # cpt_plh += (plh_out[res_star["plh_mask"]].argmax(-1) == res_star["plh_tgt"][res_star["plh_mask"]]).sum().item()
        # # print(cmb_out.transpose(1, 2)[res_star["cmb_mask"]].shape)
        # # print(res_star["cmb_tgt"][res_star["cmb_mask"]].shape)
        # cpt_cmb += (cmb_out.transpose(1, 2)[res_star["cmb_mask"]].argmax(-1) == res_star["cmb_tgt"][res_star["cmb_mask"]]).sum().item()
        # cpt_tok += (tok_out[res_star["tok_mask"]].argmax(-1) == res_star["tok_tgt"][res_star["tok_mask"]]).sum().item()
        # cpt_del_zeros += (
        #     del_out[res_star["del_mask"] & (res_star["del_tgt"] == 0)].argmax(-1) 
        #     == res_star["del_tgt"][res_star["del_mask"] & (res_star["del_tgt"] == 0)]
        # ).sum().item()
        # cpt_plh_non_zeros += (
        #     plh_out[res_star["plh_mask"] & (res_star["plh_tgt"].ne(0))].argmax(-1) 
        #     == res_star["plh_tgt"][res_star["plh_mask"] & (res_star["plh_tgt"].ne(0))]
        # ).sum().item()

        # tot_del += res_star["del_mask"].sum().item()
        # tot_plh += res_star["plh_mask"].sum().item()
        # tot_cmb += res_star["cmb_mask"].sum().item()
        # tot_tok += res_star["tok_mask"].sum().item()
        # tot_del_zeros += (res_star["del_mask"] & (res_star["del_tgt"] == 0)).sum().item()
        # tot_plh_non_zeros += (res_star["plh_mask"] & (res_star["plh_tgt"].ne(0))).sum().item()

    print()
    print("del_acc", cpt_del / tot_del, tot_del)
    print("plh_acc", cpt_plh / tot_plh, tot_plh)
    print("cmb_acc", cpt_cmb / tot_cmb, tot_cmb)
    print("tok_acc", cpt_tok / tot_tok, tot_tok)
    print("del_acc_zeros", cpt_del_zeros / tot_del_zeros, tot_del_zeros)
    print("plh_acc_non_zeros", cpt_plh_non_zeros / tot_plh_non_zeros, tot_plh_non_zeros)

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

