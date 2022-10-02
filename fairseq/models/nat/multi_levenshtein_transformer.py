# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
# from fairseq.data.multi_source_dataset import index_sentence_for_embedding
import random as rd
import numpy as np

from .levenshtein_utils import (
    _apply_del_words,
    _apply_ins_masks,
    _apply_ins_words,
    _fill as _fill_single,
    _get_del_targets,
    _get_ins_targets,
    _skip as _skip_single,
    _skip_encoder_out as _skip_encoder_out_single,
)

from .multi_levenshtein_utils import (
    pi_del,
    pi_del_single,
    pi_sel,
    pi_mask,
    pi_star,
    handle_all_plh_case,
    apply_del,
    apply_plh,
    apply_cmb,
    apply_tok,
    _skip,
    _skip_encoder_out,
    _fill,
)


@register_model("multi_lev_transformer")
class MultiLevenshteinTransformerModel(FairseqNATModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.beta = args.random_del_prob
        self.gamma = args.selection_noise_prob
        self.delta = args.completion_noise_prob
        self.eps = args.nothing_todo_plh_prob
        self.Kmax = args.plh_max_num_insert
        self.full_levt = getattr(args, "basic_levt_align", False)
        self.full_mlevt = getattr(args, "full_mlevt_align", False)
        self.max_valency = getattr(args, "max_valency", 10)
        self.curriculum_post_del_extra = args.curriculum_post_del_extra

    @property
    def allow_length_beam(self):
        return False

    @staticmethod
    def add_args(parser):
        FairseqNATModel.add_args(parser)
        parser.add_argument(
            "--early-exit",
            default="6,6,6,6",
            type=str,
            help="number of decoder layers before del, plh, cmb, tok",
        )
        parser.add_argument(
            "--no-share-discriminator",
            action="store_true",
            help="separate parameters for discriminator",
        )
        parser.add_argument(
            "--no-share-maskpredictor",
            action="store_true",
            help="separate parameters for mask-predictor",
        )
        parser.add_argument(
            "--share-discriminator-maskpredictor",
            action="store_true",
            help="share the parameters for both mask-predictor and discriminator",
        )
        parser.add_argument(
            "--sampling-for-deletion",
            action="store_true",
            help="instead of argmax, use sampling to predict the tokens",
        )
        parser.add_argument(
            "--random-del-prob",
            default=0.2,
            type=float,
            help="probability to train from noised target instead of the retrieved sequence",
        )
        # parser.add_argument(
        #     "--correction-prob",
        #     default=0.2,
        #     type=float,
        #     help="probability to train to correct errors instead",
        # )
        parser.add_argument(
            "--selection-noise-prob",
            default=0.2,
            type=float,
            help="probability to add noise individually during the selection step",
        )
        parser.add_argument(
            "--completion-noise-prob",
            default=0.2,
            type=float,
            help="probability to use the masked target to predict tokens instead",
        )
        parser.add_argument(
            "--nothing-todo-plh-prob",
            default=0.4,
            type=float,
            help="probability to use the target as prev target and predict no plh to insert",
        )
        parser.add_argument(
            "--plh-max-num-insert",
            default=64,
            type=int,
            help="Number of placeholders that can be added between 2 consecutive tokens",
        )
        parser.add_argument(
            "--curriculum-post-del-extra",
            default=-1,
            type=int,
            help="number of iterations to wait before computing post-del-extra loss",
        )
        parser.add_argument(
            "--num-retrieved",
            default=1,
            type=int,
            help="Number of sentences retrieved, then edited together to form the final sentence",
        )
        parser.add_argument(
            "--max-valency",
            default=-1,
            type=int,
            help="Clamping of the alignment algorithm complexity.",
        )
        parser.add_argument(
            "--basic-levt-align",
            action="store_true",
            help="Use the lev distance to individually align to-be-edited sequences.",
        )
        parser.add_argument(
            "--full-mlevt-align",
            action="store_true",
            help="Use the mlev alignment, even for the problematic sentences.",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = MultiLevenshteinTransformerDecoder(
            args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def regularize_shapes(self, ys, y):
        bsz = y.size(0)
        M = max(ys.size(-1), y.size(-1))
        N = ys.size(1)
        shape = (bsz, N + 1, M)
        X = y.new(*shape).fill_(self.pad)
        X[:, -1, : y.size(-1)] = y
        X[:, :-1, : ys.size(-1)] = ys

        return X[:, :-1, :], X[:, -1, :]

    @staticmethod
    def get_mask_from_prob(bsz, p):
        return torch.rand(bsz) > p

    @staticmethod
    def combine_res(res1, res2, mask):
        res = dict()
        for key in res1:
            # print(key, res1[key].shape, res2[key].shape, mask.shape)
            shape = [i for i in res1[key].shape]
            shape[0] += res2[key].size(0)
            res[key] = res1[key].new_empty(shape)
            res[key][mask] = res1[key]
            res[key][~mask] = res2[key]
        return res

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, num_iter, ids=None, **kwargs):
        # print(src_tokens.shape)
        # print(tgt_tokens.shape)
        # print(prev_output_tokens.shape)
        # print((prev_output_tokens == self.pad).sum())
        # index_seq = index_sentence_for_embedding(prev_output_tokens, self.bos)
        prev_output_tokens, tgt_tokens = self.regularize_shapes(
            prev_output_tokens, tgt_tokens
        )
        if self.full_levt:
            mask_good = torch.zeros(src_tokens.size(
                0), device=src_tokens.device, dtype=torch.bool)
        elif self.full_mlevt:
            mask_good = torch.ones(src_tokens.size(
                0), device=src_tokens.device, dtype=torch.bool)
        else:
            mask_good = torch.ones(src_tokens.size(
                0), device=src_tokens.device, dtype=torch.bool)
            threshold = src_tokens.size(1) / 3 + 10
            if threshold < src_tokens.size(1):
                for b in range(src_tokens.size(0)):
                    toks, tok_counts = tgt_tokens[b].unique(return_counts=True)
                    bad_toks = toks[tok_counts > threshold]
                    if len(bad_toks) > 0:
                        for bad_tok in bad_toks:
                            for prev_output_tokens_single in prev_output_tokens[b]:
                                if (prev_output_tokens_single == bad_tok).sum() > threshold:
                                    mask_good[b] = False
        # print("mask good", mask_good.int().cpu().tolist())
        # src_tokens_good, prev_output_tokens_good, tgt_tokens_good = src_tokens[mask_good], prev_output_tokens[mask_good], tgt_tokens[mask_good]
        # src_tokens_bad, prev_output_tokens_bad, tgt_tokens_bad = src_tokens[~mask_good], prev_output_tokens[~mask_good], tgt_tokens[~mask_good]

        # encoding
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs)

        del_tgt = None
        del_mask = None
        # choose init
        mask_star = self.get_mask_from_prob(
            prev_output_tokens.size(0), self.beta)  # for pi del
        with torch.no_grad():

            # print("max val", self.max_valency)
            # print("K max", self.Kmax)
            # print(self.pad, self.unk)

            # torch.save(
            #     prev_output_tokens[mask_star][mask_good[mask_star]].cpu(),
            #     "/linkhome/rech/genrqo01/ufn16wp/NLP4NLP/fairseq/prev_output.npy"
            # )
            # torch.save(
            #     tgt_tokens[mask_star][mask_good[mask_star]].cpu(),
            #     "/linkhome/rech/genrqo01/ufn16wp/NLP4NLP/fairseq/tgt_tokens.npy"
            # )
            # print("prev", prev_output_tokens.shape, tgt_tokens.shape)
            res_star = pi_star(
                prev_output_tokens[mask_star][mask_good[mask_star]],
                tgt_tokens[mask_star][mask_good[mask_star]],
                max_valency=self.max_valency,
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )
            # mask_debug = ~((res_star["y_cmb"] == self.eos).sum(-1).sum(-1) == res_star["y_cmb"].size(1))
            # if ids is not None and mask_debug.any().item():
            #     print("ids", ids[mask_star][mask_good[mask_star]][mask_debug])
            # print("prev", prev_output_tokens[mask_star][mask_good[mask_star]][mask_debug])
            # print("tgt", tgt_tokens[mask_star][mask_good[mask_star]][mask_debug])
            # print("y_plh", res_star["y_plh"][mask_debug])
            # print("y_cmb", res_star["y_cmb"][mask_debug])
            # print("y_tok", res_star["y_tok"][mask_debug])
            # print([(res_star["y_cmb"] == self.eos).sum(-1)])
            # print("bos = ", self.bos)
            # 19
            # print(torch.arange(mask_star.size(0), device=prev_output_tokens.device)[mask_star][mask_good[mask_star]][(res_star["y_cmb"] == self.bos).sum(-1).ne(1).any(-1)])
            # print("tgt ???", tgt_tokens[mask_star][mask_good[mask_star]][(res_star["y_cmb"] == self.bos).sum(-1).ne(1).any(-1)])
            # print("ids", ids[mask_star][mask_good[mask_star]][(res_star["y_cmb"] == self.bos).sum(-1).ne(1).any(-1)])
            # print("y_del where no bos/eos: ", (prev_output_tokens[mask_star][mask_good[mask_star]][(res_star["y_cmb"] == self.bos).sum(-1).ne(1)]).tolist())
            # print("y_plh where no bos/eos: ", (res_star["y_plh"][(res_star["y_cmb"] == self.bos).sum(-1).ne(1)]).tolist())
            # print("y_cmb where no bos/eos: ", (res_star["y_cmb"][(res_star["y_cmb"] == self.bos).sum(-1).ne(1)]).tolist())
            assert ((res_star["y_cmb"] == self.eos).sum(-1) == 1).all().item(
            ), ((res_star["y_cmb"] == self.bos).sum(-1) == 1).all().item()
            res_star["del_tgt"] = 1 - res_star["del_tgt"]
            res_star["del_tgt"][~res_star["del_mask"]] = 0
            if not mask_good[mask_star].all():
                # regular levt single alignment
                del_tgt_bad = list()
                del_mask_bad = list()
                plh_tgt_bad = list()
                plh_mask_bad = list()
                tok_mask_bad = None
                y_cmb_bad = list()
                y_plh_bad = list()
                for n in range(prev_output_tokens.size(1)):
                    # prev del levt
                    del_tgt_bad.append(_get_del_targets(
                        prev_output_tokens[mask_star][~mask_good[mask_star]][:, n],
                        tgt_tokens[mask_star][~mask_good[mask_star]],
                        self.pad,
                        prev_output_tokens.device
                    ).unsqueeze(1))
                    del_mask_bad.append(prev_output_tokens[mask_star][~mask_good[mask_star]][:, n].ne(
                        self.pad).unsqueeze(1))
                    y_plh_bad_, _, _ = _apply_del_words(
                        prev_output_tokens[mask_star][~mask_good[mask_star]][:, n],
                        in_scores=None,
                        in_attn=None,
                        word_del_pred=del_tgt_bad[-1][:, 0].bool(),
                        padding_idx=self.pad,
                        bos_idx=self.bos,
                        eos_idx=self.eos,
                    )
                    # | prev_word_del_out[~mask_good][:, n].max(-1)[1].bool(),
                    y_plh_bad.append(y_plh_bad_)
                    # delete unnecessary paddings
                    # cut_off = y_plh_bad_.ne(self.pad).sum(1).max()
                    # y_plh_bad_ = y_plh_bad_[:, :cut_off]
                    # ins levt
                    _, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                        y_plh_bad_, tgt_tokens[mask_star][~mask_good[mask_star]
                                                          ], self.pad, self.unk, y_plh_bad_.device
                    )
                    y_cmb_bad.append(masked_tgt_tokens)
                    # if tok_mask_bad is None:
                    #     tok_mask_bad = masked_tgt_masks
                    # else:
                    #     tok_mask_bad = (tok_mask_bad & masked_tgt_masks)
                    plh_tgt_bad.append(mask_ins_targets.clamp(
                        min=0, max=63).unsqueeze(1))  # for safe prediction
                    plh_mask_bad.append(
                        y_plh_bad_[:, 1:].ne(self.pad).unsqueeze(1))
                del_tgt_bad = torch.cat(del_tgt_bad, dim=1)
                del_mask_bad = torch.cat(del_mask_bad, dim=1)
                y_plh_bad = torch.cat([t.unsqueeze(1)
                                      for t in y_plh_bad], dim=1)
                y_cmb_bad = torch.cat([t.unsqueeze(1)
                                      for t in y_cmb_bad], dim=1)
                plh_tgt_bad = torch.cat(plh_tgt_bad, dim=1)
                plh_mask_bad = torch.cat(plh_mask_bad, dim=1)
                cmb_mask_bad = y_cmb_bad.ne(self.pad)
                cmb_tgt_bad = (cmb_mask_bad & y_cmb_bad.ne(self.unk))
                y_tok_bad = tgt_tokens[mask_star][~mask_good[mask_star]].clone(
                )
                y_tok_bad[(cmb_tgt_bad == self.unk).all(1)] = self.unk
                tok_mask_bad = y_tok_bad.ne(self.unk)
                cmb_tgt_bad = cmb_tgt_bad.long()

                res_star = self.combine_res(
                    {
                        "del_tgt": del_tgt_bad, "del_mask": del_mask_bad,
                        "plh_tgt": plh_tgt_bad, "plh_mask": plh_mask_bad, "y_plh": y_plh_bad,
                        "cmb_tgt": cmb_tgt_bad, "cmb_mask": cmb_mask_bad, "y_cmb": y_cmb_bad,
                        "tok_tgt": tgt_tokens[mask_star][~mask_good[mask_star]], "tok_mask": tok_mask_bad, "y_tok": y_tok_bad
                    },
                    res_star,
                    ~mask_good[mask_star]
                )
            res_del = pi_del(
                prev_output_tokens[~mask_star].shape,
                tgt_tokens[~mask_star],
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                bos_symbol=self.bos,
                eos_symbol=self.eos,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )
            res_del["del_tgt"] = 1 - res_del["del_tgt"]
            res_del["del_tgt"][~res_del["del_mask"]] = 0

            res_star = self.combine_res(res_star, res_del, mask_star)

            del_tgt = res_star["del_tgt"]
            del_mask = res_star["del_mask"]
            plh_tgt = res_star["plh_tgt"]
            plh_mask = res_star["plh_mask"]
            cmb_tgt = res_star["cmb_tgt"]
            cmb_mask = res_star["cmb_mask"]
            tok_tgt = res_star["tok_tgt"]
            tok_mask = res_star["tok_mask"]

            y_plh = res_star["y_plh"]
            y_cmb = res_star["y_cmb"]
            y_tok = res_star["y_tok"]

            y_cmb = pi_sel(
                y_cmb,
                prev_output_tokens,
                self.gamma,
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                bos_symbol=self.bos,
                eos_symbol=self.eos,
                device=src_tokens.device,
            )
            cmb_tgt = handle_all_plh_case(cmb_tgt, y_tok, y_cmb, self.unk)

            # POST PLH
            mask_not_self_target = self.get_mask_from_prob(
                prev_output_tokens.size(0), self.eps)  # for pi del
            res_post_del = pi_del_single(
                # prev_output_tokens[mask_not_self_target].shape,
                tgt_tokens[mask_not_self_target],
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                bos_symbol=self.bos,
                eos_symbol=self.eos,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )
            # print("res post plh", res_post_del)
            # res_post_del["y_plh"] = res_post_del["y_plh"][:, 0]
            # res_post_del["plh_tgt"] = res_post_del["plh_tgt"][:, 0]
            # res_post_del["plh_mask"] = res_post_del["plh_mask"][:, 0]
            res_post_del = self.combine_res(
                {
                    "plh_tgt": torch.zeros_like(tgt_tokens[~mask_not_self_target][:, 1:]),
                    "plh_mask": tgt_tokens[~mask_not_self_target][:, 1:].ne(self.pad),
                    "y_plh": tgt_tokens[~mask_not_self_target],
                },
                res_post_del,
                ~mask_not_self_target,
            )

            y_post_plh = res_post_del["y_plh"]
            post_plh_tgt = res_post_del["plh_tgt"]
            post_plh_mask = res_post_del["plh_mask"]

            mask_mask = self.get_mask_from_prob(y_tok.size(0), self.delta)
            y_tok[~mask_mask], tok_tgt[~mask_mask], tok_mask[~mask_mask] = pi_mask(
                tok_tgt[~mask_mask],
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                bos_symbol=self.bos,
                eos_symbol=self.eos,
                device=src_tokens.device,
            )

        del_out, _ = self.decoder.forward_del(
            normalize=False, prev_output_tokens=prev_output_tokens, encoder_out=encoder_out,
        )
        plh_out, _ = self.decoder.forward_plh(
            normalize=False, prev_output_tokens=y_plh, encoder_out=encoder_out,
        )
        cmb_out, _ = self.decoder.forward_cmb(
            normalize=False, prev_output_tokens=y_cmb, encoder_out=encoder_out,
        )
        cmb_out = cmb_out.transpose(1, 2)
        tok_out, _ = self.decoder.forward_tok(
            normalize=False, prev_output_tokens=y_tok, encoder_out=encoder_out,
        )

        with torch.no_grad():
            if self.decoder.sampling_for_deletion:
                y_post_del = torch.multinomial(
                    F.softmax(tok_out, -1).view(-1, tok_out.size(-1)), 1
                ).view(tok_out.size(0), -1)
            else:
                y_post_del = F.log_softmax(tok_out, dim=-1).max(2)[1]
            y_post_del.masked_scatter_(
                ~tok_mask, tgt_tokens[~tok_mask]
            )
            post_del_tgt = (y_post_del.ne(tgt_tokens)).long()
            post_del_mask = (
                y_post_del.ne(self.pad)
                & y_post_del.ne(self.bos)
                & y_post_del.ne(self.eos)
            )

        # print("y_post_del", y_post_del)
        post_del_out, _ = self.decoder.forward_del(
            normalize=False, prev_output_tokens=y_post_del, encoder_out=encoder_out,
        )
        # print("y_post_plh", y_post_plh)
        post_plh_out, _ = self.decoder.forward_plh(
            normalize=False, prev_output_tokens=y_post_plh, encoder_out=encoder_out,
        )
        if num_iter > self.curriculum_post_del_extra:
            # print("y_post_plh", y_post_plh.shape)
            with torch.no_grad():
                # apply post_plh_out to y_post_plh
                post_plh_pred = post_plh_out.max(-1)[1]
                # print("post_plh_pred", post_plh_pred.shape)
                # max_lens = torch.zeros_like(post_plh_pred).fill_(10)

                # Add a penalty by substraction in the prediction of plh
                # to ensure the sum does not get higher than 255.
                plh_penalty = torch.max(
                    post_plh_pred.max(-1)[0] * (1 - (255 - y_post_plh.ne(
                        self.pad).sum(-1)) / (post_plh_pred.sum(-1) + 1)),
                    torch.zeros_like(post_plh_pred[:, 0])
                )[:, None].expand_as(post_plh_pred)
                post_plh_pred = torch.max(
                    post_plh_pred - plh_penalty.long(),
                    torch.zeros_like(post_plh_pred)
                )
                # print("plh penalty", plh_penalty)
                # print(post_plh_pred)
                # post_plh_pred = torch.min(
                #     post_plh_pred, max_lens
                # )
                # print("y_post_plh", y_post_plh)
                y_post_tok, _ = _apply_ins_masks(
                    y_post_plh.clone(),
                    None,
                    post_plh_pred,
                    self.pad,
                    self.unk,
                    self.eos,
                )
                # print("y_post_tok", y_post_tok)
                extra_mask = y_post_tok.ne(self.unk)
                # print("y_post_tok", y_post_tok.shape)
                post_tok_out, _ = self.decoder.forward_tok(
                    normalize=False, prev_output_tokens=y_post_tok, encoder_out=encoder_out,
                )
                # print("post_tok_out", post_tok_out.shape)
                # apply post_tok_out to y_post_tok
                if self.decoder.sampling_for_deletion:
                    y_post_del_extra = torch.multinomial(
                        F.softmax(post_tok_out, -1).view(-1,
                                                         post_tok_out.size(-1)), 1
                    ).view(post_tok_out.size(0), -1)
                else:
                    y_post_del_extra = F.log_softmax(
                        post_tok_out, dim=-1).max(2)[1]
                # print("y_post_del_extra", y_post_del_extra.shape)
                # print("y_post_del", y_post_del_extra.shape)
                # print("extra_mask", extra_mask.shape)
                # print("extra_mask", extra_mask.sum(-1))
                # print("tgt_tokens", tgt_tokens.shape)
                # print("y_post_plh", y_post_plh.shape)
                y_post_del_extra.masked_scatter_(
                    extra_mask,
                    y_post_tok[extra_mask]
                )
                # print("y_post_del_extra", y_post_del_extra)
                post_del_extra_tgt = _get_del_targets(
                    y_post_del_extra,
                    tgt_tokens,
                    self.pad,
                    device=tgt_tokens.device
                )
                # print("post_del_extra_tgt", post_del_extra_tgt.shape)
                post_del_extra_mask = (
                    y_post_del_extra.ne(self.pad)
                )
            post_del_extra_out, _ = self.decoder.forward_del(
                normalize=False, prev_output_tokens=y_post_del_extra, encoder_out=encoder_out,
            )

        # print("del_tgt", del_tgt.shape)
        # print("del_mask", del_mask.shape)
        # print("del_out", del_out.shape)
        # print("plh_tgt", plh_tgt.shape)
        # print("plh_mask", plh_mask.shape)
        # print("plh_out", plh_out.shape)
        # print("cmb_tgt", cmb_tgt.shape)
        # print("cmb_mask", cmb_mask.shape)
        # print("cmb_out", cmb_out.shape)
        # print("tok_tgt", tok_tgt.shape)
        # print("tok_mask", tok_mask.shape)
        # print("tok_out", tok_out.shape)

        output = dict()

        output["prev_word_del"] = {
            "out": del_out,
            "tgt": del_tgt,
            "mask": del_mask,
            "factor": 0.5,
        }
        output["mask_ins"] = {
            "out": plh_out,
            "tgt": plh_tgt,
            "mask": plh_mask,
            "ls": 0.01,
        }
        output["cmb"] = {
            "out": cmb_out,
            "tgt": cmb_tgt,
            "mask": cmb_mask,
        }
        output["word_ins"] = {
            "out": tok_out,
            "tgt": tok_tgt,
            "mask": tok_mask,
            "ls": self.args.label_smoothing,
            "nll_loss": True,
        }
        if False:
            output["post_word_del"] = {
                "out": post_del_out.new(1, 1, 2).fill_(0),
                "tgt": post_del_tgt.new(1, 1).fill_(0),
                "mask": post_del_mask.new(1, 1).fill_(1),
                "factor": 1.0,
            }
        else:
            output["post_word_del"] = {
                "out": post_del_out,
                "tgt": post_del_tgt,
                "mask": post_del_mask,
            }
        output["post_plh"] = {
            "out": post_plh_out,
            "tgt": post_plh_tgt,
            "mask": post_plh_mask,
        }
        if num_iter > self.curriculum_post_del_extra:
            output["post_word_del_extra"] = {
                "out": post_del_extra_out,
                "tgt": post_del_extra_tgt,
                "mask": post_del_extra_mask,
            }
        else:
            output["post_word_del_extra"] = {
                "out": post_plh_out.new(1, 1, 2).fill_(0),
                "tgt": post_plh_tgt.new(1, 1).fill_(0),
                "mask": post_plh_mask.new(1, 1).fill_(1),
                "factor": 1.0,
            }

        return output

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens  # B x N x M

        if len(output_tokens.shape) == 3:
            return self.forward_decoder_multi(decoder_out, encoder_out, eos_penalty=0.0, max_ratio=max_ratio, **kwargs)
        elif len(output_tokens.shape) == 2:
            return self.forward_decoder_single(decoder_out, encoder_out, eos_penalty=eos_penalty, max_ratio=max_ratio, **kwargs)
        else:
            raise ValueError("output shape ({}) of incorrect length. only 2 and 3 acceptable.".format(
                output_tokens.shape))

    def forward_decoder_multi(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens  # B x N x M
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn.unsqueeze(-1).expand(
            decoder_out.attn.size(0),
            decoder_out.attn.size(1),
            decoder_out.attn.size(2),
            encoder_out["encoder_padding_mask"][0].size(-1),
        ).clone()
        history = decoder_out.history
        history_ops = decoder_out.history_ops

        max_lens = 255
        # verbose = 1

        # delete words
        # do not delete tokens if it is <s> </s>

        can_del_word = (output_tokens.ne(self.pad).sum(-1) > 2).any(-1)
        if can_del_word.sum() != 0:  # we cannot delete, skip

            # Skip ignores batch element with no possible deletion
            del_out, del_attn = self.decoder.forward_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_del_word),
            )
            # print("del attn not None", del_out.dtype, del_attn.dtype, )
            # del_out = F.softmax(del_out, -1)
            del_pred = del_out.max(-1)[1].bool()
            # apply_del(in_tokens, in_scores, in_attn, word_del_pred, padding_idx, bos_idx, eos_idx):
            # print("output_tokens[can_del_word]", output_tokens[can_del_word].shape)
            # print("del pred from out", del_pred.shape)
            _tokens, _scores, _attn = apply_del(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                del_attn,
                ~del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            # print(output_tokens.shape, can_del_word.shape, _tokens.shape)
            output_tokens = _fill(
                output_tokens, can_del_word, _tokens, self.pad
            )
            # print(output_scores.shape, can_del_word.shape, _scores.shape)
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            # print(attn.shape, can_del_word.shape, can_del_word.sum(), _attn.shape)
            # print("after del >>>", output_tokens.shape, output_scores.shape)
            # print(attn.dtype, _attn.dtype)
            attn = _fill(attn, can_del_word, _attn, 0.0)
            # print("attttttttttn", attn.shape)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("del", self.scatter_del(del_pred, can_del_word)))

        # insert placeholders
        can_plh = (output_tokens.ne(self.pad).sum(-1) < max_lens).any(-1)
        if can_plh.sum() != 0:
            plh_out, _ = self.decoder.forward_plh(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_plh),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_plh),
            )
            if eos_penalty > 0.0:
                plh_out[:, :, :, 0] = plh_out[:, :, :, 0] - eos_penalty
            # plh_out = F.softmax(plh_out, -1)
            plh_pred = plh_out.max(-1)[1]
            plh_pred = torch.minimum(
                plh_pred,
                torch.tensor(255, device=plh_pred.device,
                             dtype=plh_pred.dtype),
            )

            _tokens, _scores = apply_plh(
                output_tokens[can_plh],
                output_scores[can_plh],
                plh_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            output_tokens = _fill(output_tokens, can_plh, _tokens, self.pad)
            # print(output_scores.shape, can_plh.shape, _scores.shape)
            output_scores = _fill(output_scores, can_plh, _scores, 0)
            # print("after plh >>>", output_tokens.shape, output_scores.shape)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("plh", self.scatter_plh(plh_pred, can_plh)))

        cmb_len = (output_tokens == self.eos).long(
        ).argsort(-1)[:, :, -1].max(-1)[0]
        mask_inf = (
            (
                torch.arange(output_tokens.size(-1),
                             device=output_tokens.device)
                .view(1, -1)
                .expand(len(cmb_len), -1)
                < cmb_len.view(-1, 1).expand(-1, output_tokens.size(-1))
            )
            .view(len(cmb_len), 1, output_tokens.size(-1))
            .expand(-1, output_tokens.size(1), -1)
        )
        mask_eq = (
            (
                torch.arange(output_tokens.size(-1),
                             device=output_tokens.device)
                .view(1, -1)
                .expand(len(cmb_len), -1)
                == cmb_len.view(-1, 1).expand(-1, output_tokens.size(-1))
            )
            .view(len(cmb_len), 1, output_tokens.size(-1))
            .expand(-1, output_tokens.size(1), -1)
        )
        output_tokens[mask_eq] = self.eos
        output_tokens[
            mask_inf & ((output_tokens == self.eos) |
                        (output_tokens == self.pad))
        ] = self.unk  # makes sense I guess

        # merge sequences
        cmb_out, _ = self.decoder.forward_cmb(
            normalize=True, prev_output_tokens=output_tokens, encoder_out=encoder_out,
        )
        # plh_out = F.softmax(plh_out, -1)
        cmb_pred = cmb_out[:, :, :, 1]

        output_tokens, output_scores = apply_cmb(
            output_tokens,
            output_scores,
            cmb_pred,
            self.pad,
            self.bos,
            self.eos,
            self.unk,
        )
        # print("after cmb >>>", output_tokens.shape, output_scores.shape)
        attn = attn[:, 0]
        if history is not None:
            history.append(output_tokens.clone())
        if history_ops is not None:
            history_ops.append(("cmb", cmb_pred))

        # insert tok
        can_tok = output_tokens.eq(self.unk).sum(1) > 0
        if can_tok.sum() != 0:
            # print("before tok >>>", _skip(output_tokens, can_tok).cpu().numpy().tolist())
            tok_out, tok_attn = self.decoder.forward_tok(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_tok),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_tok
                ),
            )
            tok_score, tok_pred = tok_out.max(-1)

            _tokens, _scores = apply_tok(
                output_tokens[can_tok],
                output_scores[can_tok],
                tok_pred,
                tok_score,
                self.unk,
            )

            output_tokens = _fill_single(
                output_tokens, can_tok, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_tok, _scores, 0)
            # print(attn.shape, can_tok.shape, tok_attn.shape, tok_out.shape, can_tok.sum())
            attn = _fill_single(attn, can_tok, tok_attn, 0.0)

            # print("after tok >>>", output_tokens.shape, output_scores.shape)

            if history is not None:
                history.append(output_tokens.clone())
        # if history_ops is not None:
        #     history_ops.append(can_tok.clone() if can_tok.sum() != 0 else None)
        #     history_ops.append(tok_pred.clone() if can_tok.sum() != 0 else None)

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(-1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn[:, :cut_off, :],
            history=history,
        )

    def forward_decoder_single(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):
        assert eos_penalty > 0.1

        output_tokens = decoder_out.output_tokens  # B x M
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history
        history_ops = decoder_out.history_ops

        max_lens = 255

        # delete words
        # do not delete tokens if it is <s> </s>
        can_del_word = output_tokens.ne(self.pad).sum(1) > 2

        if can_del_word.sum() != 0:  # we cannot delete, skip

            # Skip ignores batch element with no possible deletion
            del_out, del_attn = self.decoder.forward_del(
                normalize=True,
                prev_output_tokens=_skip_single(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out_single(
                    self.encoder, encoder_out, can_del_word),
            )
            del_pred = del_out.max(-1)[1].bool()

            _tokens, _scores, _attn = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                del_attn,
                del_pred,
                self.pad,
                self.bos,
                self.eos,
            )
            output_tokens = _fill_single(
                output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill_single(
                output_scores, can_del_word, _scores, 0)
            attn = _fill_single(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("del", self.scatter_del(del_pred, can_del_word)))

        # print("after del", output_tokens.shape, output_scores.shape)
        # insert placeholders
        can_plh = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_plh.sum() != 0:
            plh_out, _ = self.decoder.forward_plh(
                normalize=True,
                prev_output_tokens=_skip_single(output_tokens, can_plh),
                encoder_out=_skip_encoder_out_single(
                    self.encoder, encoder_out, can_plh),
            )
            if eos_penalty > 0.0:
                plh_out[:, :, 0] = plh_out[:, :, 0] - eos_penalty
            plh_pred = plh_out.max(-1)[1]
            plh_pred = torch.minimum(
                plh_pred,
                torch.tensor(255, device=plh_pred.device,
                             dtype=plh_pred.dtype),
            )

            _tokens, _scores = _apply_ins_masks(
                output_tokens[can_plh],
                output_scores[can_plh],
                plh_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            # print("special ones", _tokens.shape, _scores.shape)
            output_tokens = _fill_single(
                output_tokens, can_plh, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_plh, _scores, 0)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("plh", self.scatter_plh(plh_pred, can_plh)))

        # print("after plh", output_tokens.shape, output_scores.shape)
        # insert tok
        can_tok = output_tokens.eq(self.unk).sum(1) > 0
        if can_tok.sum() != 0:
            tok_out, tok_attn = self.decoder.forward_tok(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_tok),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_tok),
            )
            tok_score, tok_pred = tok_out.max(-1)
            _tokens, _scores = _apply_ins_words(
                output_tokens[can_tok],
                output_scores[can_tok],
                tok_pred,
                tok_score,
                self.unk,
            )
            output_tokens = _fill_single(
                output_tokens, can_tok, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_tok, _scores, 0)
            attn = _fill_single(attn, can_tok, tok_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
        # if history_ops is not None:
        #     history_ops.append(can_tok.clone() if can_tok.sum() != 0 else None)
        #     history_ops.append(tok_pred.clone() if can_tok.sum() != 0 else None)

        # print("after tok", output_tokens.shape, output_scores.shape)
        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(-1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        # print("outoutoutoutout", output_tokens.shape, output_scores.shape)
        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=attn,
            history=history,
        )

    def scatter_del(self, pred, can):
        shape = list(pred.shape)
        shape[0] = can.size(0)
        pred_scattered = pred.new(*shape).fill_(0)
        pred_scattered[can] = pred
        return pred_scattered

    def scatter_plh(self, pred, can):
        shape = list(pred.shape)
        shape[0] = can.size(0)
        pred_scattered = pred.new(*shape).fill_(0)
        pred_scattered[can] = pred
        return pred_scattered

    def initialize_output_tokens(self, encoder_out, multi_src_tokens):

        return DecoderOut(
            output_tokens=multi_src_tokens,
            output_scores=torch.zeros_like(
                multi_src_tokens
            ).type_as(encoder_out["encoder_out"][0]),
            attn=torch.zeros_like(
                multi_src_tokens, dtype=torch.float32
            ),
            step=0,
            max_step=0,
            history=None,
            history_ops=None
        )

    def _initialize_output_tokens(self, encoder_out, src_tokens, multi_src_tokens):
        initial_output_tokens = src_tokens.new_zeros(src_tokens.size(0), 2)
        initial_output_tokens[:, 0] = self.bos
        initial_output_tokens[:, 1] = self.eos

        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out["encoder_out"][0])

        return DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            attn=None,
            step=0,
            max_step=0,
            history=None,
            history_ops=None
        )


class MultiLevenshteinTransformerDecoder(FairseqNATDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.dictionary = dictionary
        self.bos = dictionary.bos()
        self.unk = dictionary.unk()
        self.eos = dictionary.eos()
        self.pad = dictionary.pad()
        self.sampling_for_deletion = getattr(
            args, "sampling_for_deletion", False)
        self.Kmax = 64
        self.embed_plh = Embedding(self.Kmax, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)
        self.embed_cmb = nn.Linear(self.output_embed_dim, 1)
        self.embed_tok = nn.Linear(self.output_embed_dim, len(self.dictionary))
        self.num_retrieved = args.num_retrieved
        self.embed_seq_num = Embedding(
            self.num_retrieved + 1, embed_tokens.embedding_dim, None
        )
        self.squash_multi_toks = True

        # del_word, ins_mask, ins_word
        self.early_exit = [int(i) for i in args.early_exit.split(",")]
        assert len(self.early_exit) == 4

        # copy layers for mask-predict/deletion
        self.layers_del = None
        if getattr(args, "no_share_discriminator", False):
            self.layers_del = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[0])
                ]
            )
        self.layers_plh = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_plh = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[1])
                ]
            )
        self.layers_cmb = None
        if getattr(args, "no_share_maskpredictor", False):
            self.layers_cmb = nn.ModuleList(
                [
                    TransformerDecoderLayer(args, no_encoder_attn)
                    for _ in range(self.early_exit[2])
                ]
            )
        if getattr(args, "share_discriminator_maskpredictor", False):
            assert getattr(
                args, "no_share_discriminator", False
            ), "must set saperate discriminator"
            self.layers_msk = self.layers_del

    def multi_squash(self, multi_toks):
        # print(multi_toks.ne(self.pad).shape)
        # print(multi_toks.ne(self.pad).shape)
        mask = multi_toks.ne(self.pad)
        # print("mask", mask.long().cpu().numpy())
        max_length = mask.view(multi_toks.size(0), -1).sum(-1).max()
        seq_index = (multi_toks == self.bos).cumsum(-1)
        squashed_toks = torch.full(
            (multi_toks.size(0), max_length),
            self.pad, dtype=multi_toks.dtype, device=multi_toks.device
        )
        sorted_ = mask.view(multi_toks.size(0), -1).long().sort(descending=True, stable=True, dim=-1)
        flat_multi_toks = multi_toks.view(multi_toks.size(0), -1)
        # new_index = sorted_[1][:, :max_length]
        squashed_toks = flat_multi_toks[
            torch.arange(
                multi_toks.size(0),
                device=multi_toks.device
            )[:, None].expand_as(flat_multi_toks), 
            sorted_[1]
        ][:, :max_length]
        # print(seq_index.shape, multi_toks.shape)
        # print(seq_index)
        return squashed_toks, seq_index, sorted_[1]

    def multi_unsquash(self, squashed, flat_index, N, L):
        # print("squashed", squashed.shape)
        # print("flat_index", flat_index.shape, "\n", flat_index.cpu().numpy())
        # print("N, L", N, L)
        multi = squashed.new_zeros(
            squashed.size(0), N * L, squashed.size(-1)
        )
        multi[
            torch.arange(
                squashed.size(0),
                device=squashed.device
            )[:, None].expand_as(flat_index[:, :squashed.size(1)]), 
            flat_index[:, :squashed.size(1)]
        ] = squashed
        # multi = squashed[
        #     torch.arange(
        #         squashed.size(0),
        #         device=squashed.device
        #     )[:, None].expand_as(flat_index), 
        #     flat_index
        # ][:, :squashed.size(1)]
        return multi.view(squashed.size(0), N, L, -1)

    def extract_features_multi(
        self,
        prev_output_tokens,
        encoder_out=None,
        early_exit=None,
        layers=None,
        multi_len=False,
        pad_to_same=False,
        **unused
    ):
        """
        Similar to *forward* but only return features.
        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: a dictionary of hidden states and masks

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        extra = dict()
        # prev_output_tokens: batch x N x M
        if multi_len and len(prev_output_tokens.shape) > 1:
            shape_multi = prev_output_tokens.shape
            
            if self.squash_multi_toks:
                prev_output_squashed, embed_index, flat_index = self.multi_squash(prev_output_tokens)
                extra.update(
                    {"N": shape_multi[1], "L": shape_multi[2], "flat_index": flat_index}
                )
            if self.embed_positions is not None:
                if self.squash_multi_toks:
                    positions_ = self.embed_positions(prev_output_tokens[:, 0, :])
                    positions_ = positions_.unsqueeze(1).expand(
                        (prev_output_tokens.size(0),
                        prev_output_tokens.size(1),
                        prev_output_tokens.size(2),
                        positions_.size(-1))
                    ).reshape(
                        (prev_output_tokens.size(0),
                        prev_output_tokens.size(1) *
                        prev_output_tokens.size(2),
                        positions_.size(-1))
                    )
                    positions = positions_[
                        torch.arange(
                            prev_output_squashed.size(0),
                            device=prev_output_squashed.device
                        )[:, None].expand_as(flat_index), 
                        flat_index
                    ][:, :prev_output_squashed.size(1)]
                else:
                    positions = self.embed_positions(prev_output_tokens[:, 0, :])
                    positions = positions.unsqueeze(1).expand(
                        (prev_output_tokens.size(0),
                        prev_output_tokens.size(1),
                        prev_output_tokens.size(2),
                        positions.size(-1))
                    )
            else:
                positions = None

            if self.squash_multi_toks:
                prev_output_tokens = prev_output_squashed
                seq_emb = self.embed_seq_num(
                    (prev_output_squashed == self.bos).cumsum(-1) - 1
                )
            else:
                seq_emb = self.embed_seq_num(
                    torch.arange(
                        prev_output_tokens.size(1), device=prev_output_tokens.device
                    )
                )
                seq_emb = seq_emb.unsqueeze(0).repeat(
                    prev_output_tokens.size(0), 1, 1)
                seq_emb = seq_emb.unsqueeze(2).repeat(
                    1, 1, prev_output_tokens.size(2), 1)
            

            tok_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)
            # print("tok embed", tok_emb.shape)
            # print("seq embed", seq_emb.shape)
            # print("prev_output_squashed", prev_output_squashed.shape)
            # print("positions", positions.shape)

            # print("tok_emb", tok_emb)
            # print("seq_emb", seq_emb)
            # print("positions", positions)

            # change shape (batch x N x M x p) to (batch x NM x p)
            # print(positions.shape, prev_output_tokens.shape, self.embed_seq_num.embedding_dim)

            if not self.squash_multi_toks:
                positions = positions.reshape(
                    prev_output_tokens.size(0),
                    prev_output_tokens.size(1) * prev_output_tokens.size(2),
                    self.embed_seq_num.embedding_dim,
                )
                seq_emb = seq_emb.reshape(
                    prev_output_tokens.size(0),
                    prev_output_tokens.size(1) * prev_output_tokens.size(2),
                    self.embed_seq_num.embedding_dim,
                )
                tok_emb = tok_emb.reshape(
                    prev_output_tokens.size(0),
                    prev_output_tokens.size(1) * prev_output_tokens.size(2),
                    self.embed_seq_num.embedding_dim,
                )

        else:
            # embed positions
            positions = (
                self.embed_positions(prev_output_tokens)
                if self.embed_positions is not None
                else None
            )
            seq_emb = self.embed_seq_num(
                torch.tensor(self.embed_seq_num.num_embeddings - 1).to(
                    prev_output_tokens.device
                )
            )
            # embed tokens and positions
            tok_emb = self.embed_scale * self.embed_tokens(prev_output_tokens)

        x = tok_emb

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
        if seq_emb is not None:
            x += seq_emb
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]

        # decoder layers
        decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for _, layer in enumerate(layers[:early_exit]):
            x, attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask.view(
                    decoder_padding_mask.size(0), -1
                ),
            )
            # print(x.dtype, attn.dtype)
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if multi_len and not self.squash_multi_toks:
            shape = (
                prev_output_tokens.size(0),
                prev_output_tokens.size(1),
                prev_output_tokens.size(2),
                -1,
            )
            x = x.view(shape)
            if attn is not None:
                attn = attn.view(shape)
        extra.update(
            {"attn": attn, "inner_states": inner_states}
        )
        return x, extra

    @ensemble_decoder
    def forward_del(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            multi_len=(len(prev_output_tokens.shape) == 3),
            **unused
        )
        # features: batch x N x M x d
        # print("features", features)
        decoder_out = F.linear(features, self.embed_word_del.weight)

        if "flat_index" in extra:
            # print("(del) N, L =", extra["N"], extra["L"])
            decoder_out = self.multi_unsquash(
                decoder_out, extra["flat_index"], extra["N"], extra["L"]
            )
            if extra["attn"] is not None:
                extra["attn"] = self.multi_unsquash(
                    extra["attn"], extra["flat_index"], extra["N"], extra["L"]
                )
        # print("forward del ok ", len(prev_output_tokens.shape) == 3, "flat_index" in extra)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_plh(self, normalize, encoder_out, prev_output_tokens, **unused):
        multi_len = (len(prev_output_tokens.shape) == 3)
        # print("prev out tok", prev_output_tokens.shape)
        # print("multi", )
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_plh,
            multi_len=multi_len,
            **unused
        )
        # features: batch x N x L x d
        # features: batch x M x d
        if "flat_index" in extra:
            # print("N, L =", extra["N"], extra["L"])
            # print("features before =", features.shape)
            features = self.multi_unsquash(
                features, extra["flat_index"], extra["N"], extra["L"]
            )
            if extra["attn"] is not None:
                extra["attn"] = self.multi_unsquash(
                    extra["attn"], extra["flat_index"], extra["N"], extra["L"]
                )
        if multi_len:
            # print("features after =", features.shape)
            features_cat = torch.cat(
                [features[:, :, :-1, :], features[:, :, 1:, :]],
                -1
            )
            # print("features cat =", features_cat.shape)
        else:
            features_cat = torch.cat(
                [features[:, :-1, :], features[:, 1:, :]],
                -1
            )
        decoder_out = F.linear(features_cat, self.embed_plh.weight)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        # print("forward plh ok ", len(prev_output_tokens.shape) == 3, "flat_index" in extra)
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_cmb(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[1],
            layers=self.layers_cmb,
            multi_len=(len(prev_output_tokens.shape) == 3),
            **unused
        )
        # features: batch x N x L x d
        # features: batch x M x d
        decoder_out = self.embed_cmb(features)
        if "flat_index" in extra:
            decoder_out = self.multi_unsquash(
                decoder_out, extra["flat_index"], extra["N"], extra["L"]
            )
            if extra["attn"] is not None:
                extra["attn"] = self.multi_unsquash(
                    extra["attn"], extra["flat_index"], extra["N"], extra["L"]
                )
        # batch x N x L
        decoder_out = (
            decoder_out.transpose(1, 2).squeeze(-1)
        )  # batch x L x N
        decoder_out = torch.sigmoid(decoder_out)
        decoder_out = torch.stack(
            (decoder_out, 1 - decoder_out), dim=-1
        )  # batch x L x N x 2

        # decoder_out: batch x L x N
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        # print("forward cmb ok ", len(prev_output_tokens.shape) == 3, "flat_index" in extra)
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_tok(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        # features: batch x L x d
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]


@register_model_architecture(
    "multi_lev_transformer", "multi_levenshtein_transformer_base"
)
def multi_levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(
        args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(
        args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(
        args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(
        args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(
        args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(
        args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(
        args, "share_discriminator_maskpredictor", False
    )
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)


@register_model_architecture("multi_lev_transformer", "multi_levenshtein_transformer")
def multi_levenshtein_transformer_wmt_en_de(args):
    multi_levenshtein_base_architecture(args)


# similar parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture(
    "multi_lev_transformer", "multi_levenshtein_transformer_vaswani_big"
)
def multi_levenshtein_transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    multi_levenshtein_base_architecture(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture(
    "multi_lev_transformer", "multi_levenshtein_transformer_big"
)
def multi_levenshtein_transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(
        args, "encoder_normalize_before", True)
    args.decoder_normalize_before = getattr(
        args, "decoder_normalize_before", True)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.1)
    multi_levenshtein_transformer_vaswani_wmt_en_de_big(args)
