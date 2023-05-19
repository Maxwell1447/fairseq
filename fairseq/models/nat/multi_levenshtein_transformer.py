# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.iterative_refinement_generator import DecoderOut
from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat import FairseqNATDecoder, FairseqNATModel, ensemble_decoder
from fairseq.models.transformer import Embedding, TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
import sys
import time
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
    realign_dp_malign,
    realign_grad_descent,
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
        self.use_insert_distribution = getattr(args, "continuous_insert", False)
        self.insert_var = getattr(args, "insert_var", 4.0)
        self.full_levt = getattr(args, "basic_levt_align", False)
        self.full_mlevt = getattr(args, "full_mlevt_align", False)
        self.max_valency = getattr(args, "max_valency", 10)
        self.curriculum_post_del_extra = args.curriculum_post_del_extra
        self.clf_num = getattr(args, "clf_num", 1)
        if self.use_insert_distribution:
            self.insert_distribution = torch.distributions.normal.Normal
            self.insert_no_distribution = torch.distributions.exponential.Exponential(rate=torch.tensor(2.))
            self.MSE = torch.nn.MSELoss()
            self.BCE = torch.nn.BCEWithLogitsLoss()

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
        parser.add_argument(
            "--unsquash",
            action="store_true",
            help="Doesn't squash the multi tokens while passing into the transformer decoder.",
        )
        parser.add_argument(
            "--continuous-insert",
            action="store_true",
            help="Change the insertion module from classification to a parametrized distribution..",
        )
        parser.add_argument(
            "--insert-var",
            type=float,
            help="Variance of the chosen distribution.",
        )
        parser.add_argument(
            "--clf-num",
            type=int,
            help="Number of classes starting from 0 where the insertion of placeholders is considered as a classification problem.",
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
        shape_n = (bsz, N, M)
        shape = (bsz, M)
        Xs = ys.new(*shape_n).fill_(self.pad)
        X = y.new(*shape).fill_(self.pad)
        Xs[:, :, :ys.size(-1)] = ys
        X[:, :y.size(-1)] = y

        return Xs, X

    @staticmethod
    def get_mask_from_prob(bsz, p):
        return torch.rand(bsz) > p

    @staticmethod
    def combine_res(res1, res2, mask):
        res = dict()
        for key in res1:
            shape = [i for i in res1[key].shape]
            shape[0] += res2[key].size(0)
            res[key] = res1[key].new_empty(shape)
            res[key][mask] = res1[key]
            res[key][~mask] = res2[key]
        return res

    def distribution_loss(self, plh_out, plh_tgt, plh_mask):
        plh_out = plh_out[plh_mask]
        plh_tgt = plh_tgt[plh_mask]

        logits = F.log_softmax(plh_out[..., :-1], dim=-1)
        loss_clf = F.nll_loss(
            logits.float(),
            plh_tgt.clamp(0, self.clf_num),
            reduction="none"
        ).mean().to(plh_out.dtype)

        mask_ins = plh_tgt >= self.clf_num
        
        if mask_ins.any():
            loss_mu = 0.001 * self.MSE(plh_out[..., -1][mask_ins], plh_tgt[mask_ins].to(plh_out.dtype))
        else:
            loss_mu = 0

        return loss_mu + loss_clf

    def predict_plh_from_out(self, plh_out, penalty=0.0, max_insert=128):
        plh_out[..., 0] -= penalty
        pred = plh_out[..., :-1].argmax(-1)
        insert_mask = (pred == (self.clf_num - 1))
        pred[insert_mask] = torch.clamp(torch.round(plh_out[..., -1][insert_mask]).long(), 0, max_insert)

        return pred


    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, num_iter, ids=None, **kwargs):
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
            # filter too complicated multi-sequences
            # they are instead aligned with levenshtein distance independently
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

        # encoding
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, **kwargs)

        del_tgt = None
        del_mask = None
        # choose init
        mask_star = self.get_mask_from_prob(
            prev_output_tokens.size(0), self.beta)  # for pi del
        with torch.no_grad():
            res_star = pi_star(
                prev_output_tokens[mask_star][mask_good[mask_star]],
                tgt_tokens[mask_star][mask_good[mask_star]],
                max_valency=self.max_valency,
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )

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
                    y_plh_bad_, _, _, _ = _apply_del_words(
                        prev_output_tokens[mask_star][~mask_good[mask_star]][:, n],
                        in_scores=None,
                        in_attn=None,
                        word_del_pred=del_tgt_bad[-1][:, 0].bool(),
                        padding_idx=self.pad,
                        bos_idx=self.bos,
                        eos_idx=self.eos,
                    )
                    y_plh_bad.append(y_plh_bad_)
                    # delete unnecessary paddings
                    _, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                        y_plh_bad_, tgt_tokens[mask_star][~mask_good[mask_star]
                                                          ], self.pad, self.unk, y_plh_bad_.device
                    )
                    y_cmb_bad.append(masked_tgt_tokens)
                    plh_tgt_bad.append(mask_ins_targets.clamp(
                        min=0, max=self.Kmax - 1).unsqueeze(1))  # for safe prediction
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
                tok_mask_bad = y_tok_bad == self.unk
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
                tgt_tokens[mask_not_self_target],
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                bos_symbol=self.bos,
                eos_symbol=self.eos,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )
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

        post_del_out, _ = self.decoder.forward_del(
            normalize=False, prev_output_tokens=y_post_del, encoder_out=encoder_out,
        )
        post_plh_out, _ = self.decoder.forward_plh(
            normalize=False, prev_output_tokens=y_post_plh, encoder_out=encoder_out,
        )
        with torch.no_grad():
            # apply post_plh_out to y_post_plh
            if self.use_insert_distribution:
                # post_plh_pred = torch.round(post_plh_out)
                post_plh_pred = self.predict_plh_from_out(post_plh_out)
            else:
                post_plh_pred = post_plh_out.max(-1)[1]

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

            y_post_tok, _, _ = _apply_ins_masks(
                y_post_plh.clone(),
                None,
                post_plh_pred,
                self.pad,
                self.unk,
                self.eos,
            )
            extra_mask = y_post_tok.ne(self.unk)
            post_tok_out, _ = self.decoder.forward_tok(
                normalize=False, prev_output_tokens=y_post_tok, encoder_out=encoder_out,
            )
            # apply post_tok_out to y_post_tok
            if self.decoder.sampling_for_deletion:
                y_post_del_extra = torch.multinomial(
                    F.softmax(post_tok_out, -1).view(-1,
                                                        post_tok_out.size(-1)), 1
                ).view(post_tok_out.size(0), -1)
            else:
                y_post_del_extra = F.log_softmax(
                    post_tok_out, dim=-1).max(2)[1]
            y_post_del_extra.masked_scatter_(
                extra_mask,
                y_post_tok[extra_mask]
            )
            post_del_extra_tgt = _get_del_targets(
                y_post_del_extra,
                tgt_tokens,
                self.pad,
                device=tgt_tokens.device
            )
            post_del_extra_mask = (
                y_post_del_extra.ne(self.pad)
            )
        post_del_extra_out, _ = self.decoder.forward_del(
            normalize=False, prev_output_tokens=y_post_del_extra, encoder_out=encoder_out,
        )

        output = dict()

        output["prev_word_del"] = {
            "out": del_out,
            "tgt": del_tgt,
            "mask": del_mask,
            "factor": 0.5,
        }
        if self.use_insert_distribution:
            output["mask_ins"] = {
                "loss": self.distribution_loss(plh_out, plh_tgt, plh_mask),
                "factor": 1
            }
        else:
            output["mask_ins"] = {
                "out": plh_out,
                "tgt": plh_tgt,
                "mask": plh_mask,
                "ls": 0.2, # original = 0.01
                "ls-type": "binomial" # binomial
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
        output["post_word_del"] = {
            "out": post_del_out,
            "tgt": post_del_tgt,
            "mask": post_del_mask,
        }
        if self.use_insert_distribution:
            output["post_plh"] = {
                "loss": self.distribution_loss(post_plh_out, post_plh_tgt, post_plh_mask),
                "factor": 1
            }
        else:
            output["post_plh"] = {
                "out": post_plh_out,
                "tgt": post_plh_tgt,
                "mask": post_plh_mask,
                "ls": 0.2, # original = 0.0
                "ls-type": "binomial" # binomial
            }
        output["post_word_del_extra"] = {
            "out": post_del_extra_out,
            "tgt": post_del_extra_tgt,
            "mask": post_del_extra_mask,
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
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, realigner="no", **kwargs
    ):

        output_tokens = decoder_out.output_tokens  # B x N x M
        output_scores = decoder_out.output_scores
        output_origin = decoder_out.output_origin
        attn = decoder_out.attn.unsqueeze(-1).expand(
            decoder_out.attn.size(0),
            decoder_out.attn.size(1),
            decoder_out.attn.size(2),
            encoder_out["encoder_padding_mask"][0].size(-1),
        ).clone()
        history = decoder_out.history
        history_ops = decoder_out.history_ops

        max_lens = 255
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
            del_pred = del_out.max(-1)[1].bool()
            _tokens, _scores, _attn, _origin = apply_del(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                del_attn,
                ~del_pred,
                self.pad,
                self.bos,
                self.eos,
                in_origin=output_origin[can_del_word]
                if output_origin is not None
                else None
            )
            output_tokens = _fill(
                output_tokens, can_del_word, _tokens, self.pad
            )
            output_scores = _fill(output_scores, can_del_word, _scores, 0)
            output_origin = _fill(output_origin, can_del_word, _origin, 0)
            attn = _fill(attn, can_del_word, _attn, 0.0)

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
            if eos_penalty > 0.0: # TODO: adapt to distro-plh
                plh_out[..., 0] = plh_out[..., 0] - eos_penalty
            output_tokens_to_plh = _skip(output_tokens, can_plh)
            if plh_out.size(1) > 1:
                if realigner == "dp_malign":
                    plh_pred_success, success_mask = realign_dp_malign(_skip(output_tokens, can_plh), plh_out, eos=self.eos)
                elif realigner == "grad_descent":
                    success_mask = torch.ones(can_plh.sum(), device=plh_out.device, dtype=bool)
                    plh_pred_success = realign_grad_descent(output_tokens_to_plh, plh_out, eos=self.eos)
                elif realigner == "grad_descent_multinomial":
                    if self.use_insert_distribution:
                        # TODO: adapt to clf/continuous mix
                        #### OLD
                        # plh_out_ = self.insert_distribution(
                        #     loc=plh_out.squeeze(-1),
                        #     scale=torch.full_like(plh_out, self.insert_var).squeeze(-1)
                        # ).log_prob(torch.arange(self.Kmax * 2))
                        #### NEW
                        # clf_penalty = plh_out[..., -2]
                        prob_clf = torch.softmax(plh_out[..., :-1], -1)[..., -1][..., None]
                        # continuous_penalty = torch.log(1 - torch.exp(clf_penalty)) # make numerically stable
                        clf_penalty = torch.log(prob_clf)
                        continuous_penalty = torch.log(1 - prob_clf)
                        plh_out_ = self.insert_distribution(
                            loc=plh_out[..., -1].squeeze(-1),
                            scale=torch.full_like(plh_out[..., -1], self.insert_var).squeeze(-1)
                        ).log_prob(torch.arange(self.Kmax))
                        plh_out_[..., self.num_clf:] = torch.log_softmax(plh_out_[..., self.num_clf:], -1) - continuous_penalty
                        # mask_clf = plh_out.argmax(-1) < self.clf_num
                        # prob_clf = torch.log_softmax(plh_out[..., :-1], -1)[..., -1][..., None]
                        # plh_out_[mask_clf] = plh_out[mask_clf]
                        plh_out_[..., :self.num_clf] = clf_penalty + torch.log_softmax(plh_out[..., :-2], -1)
                        # no upper limit
                        # TODO: adapt realign_grad_descent() to handle the distribution
                    else:
                        plh_out_ = plh_out
                    plh_pred_success = realign_grad_descent(
                        output_tokens_to_plh,
                        plh_out_,
                        bos=self.bos,
                        pad=self.pad,
                        eos=self.eos,
                        unk=self.unk,
                        max_dist=5.0,
                        lr=0.006,
                        momentum=0.97,
                        scheduler_sqrt_rate=0.36,
                        num_iter=100,
                        alpha=0.35,
                        gamma=0.65,
                        start=0.25,
                        end=0.9,
                        len_loss_scale=0.78,
                        p=2,
                        log_prob_loss_type="multinomial_pdf",
                        sigma=1.0,
                        tau=1.0
                    )
                else:
                    success_mask = torch.zeros(can_plh.sum(), device=plh_out.device, dtype=bool)
            else:
                success_mask = torch.zeros(can_plh.sum(), device=plh_out.device, dtype=bool)


            ##### PLH PRED
            if self.use_insert_distribution:
                plh_pred = torch.zeros_like(plh_out[..., -1], dtype=torch.long)
                # plh_pred[~success_mask] = torch.round(plh_out[~success_mask].squeeze(-1))
                plh_pred[~success_mask] = self.predict_plh_from_out(plh_out[~success_mask])
                if success_mask.any():
                    plh_pred[success_mask] = plh_pred_success[success_mask]
            else:
                plh_pred = torch.zeros_like(plh_out[..., -1], dtype=torch.long)
                plh_pred[~success_mask] = plh_out[~success_mask].max(-1)[1]
                if success_mask.any():
                    plh_pred[success_mask] = plh_pred_success[success_mask]

            plh_pred = torch.minimum(
                plh_pred,
                torch.tensor(255, device=plh_pred.device,
                            dtype=plh_pred.dtype),
            )
            _tokens, _scores, _origin = apply_plh(
                output_tokens[can_plh],
                output_scores[can_plh],
                plh_pred,
                self.pad,
                self.unk,
                self.eos,
                in_origin=output_origin[can_plh]
                if output_origin is not None
                else None
            )
            output_tokens = _fill(output_tokens, can_plh, _tokens, self.pad)
            output_scores = _fill(output_scores, can_plh, _scores, 0)
            output_origin = _fill(output_origin, can_plh, _origin, 0)

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
        ] = self.unk

        # merge sequences
        cmb_out, _ = self.decoder.forward_cmb(
            normalize=True, prev_output_tokens=output_tokens, encoder_out=encoder_out,
        )
        cmb_pred = F.log_softmax(cmb_out, -1)[..., 1]

        output_tokens, output_scores, output_origin = apply_cmb(
            output_tokens,
            output_scores,
            cmb_pred,
            self.pad,
            self.bos,
            self.eos,
            self.unk,
            in_origin=output_origin
        )
        attn = attn[:, 0]
        if history is not None:
            history.append(output_tokens.clone())
        if history_ops is not None:
            history_ops.append(("cmb", cmb_pred))

        # insert tok
        can_tok = output_tokens.eq(self.unk).sum(1) > 0
        if can_tok.sum() != 0:
            tok_out, tok_attn = self.decoder.forward_tok(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_tok),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_tok
                ),
            )
            tok_score, tok_pred = tok_out.max(-1)

            _tokens, _scores, _origin = apply_tok(
                output_tokens[can_tok],
                output_scores[can_tok],
                tok_pred,
                tok_score,
                self.unk,
                in_origin=output_origin[can_tok]
                if output_origin is not None
                else None
            )

            output_tokens = _fill_single(
                output_tokens, can_tok, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_tok, _scores, 0)
            output_origin = _fill_single(output_origin, can_tok, _origin, 0)
            attn = _fill_single(attn, can_tok, tok_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(-1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        if output_origin is not None:
            output_origin = output_origin[:, :cut_off]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            output_origin=output_origin,
            attn=attn[:, :cut_off, :],
            history=history,
        )

    def forward_decoder_single(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens  # B x M
        output_scores = decoder_out.output_scores
        attn = decoder_out.attn
        history = decoder_out.history
        history_ops = decoder_out.history_ops
        output_origin = decoder_out.output_origin

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

            _tokens, _scores, _attn, _origin = _apply_del_words(
                output_tokens[can_del_word],
                output_scores[can_del_word],
                del_attn,
                del_pred,
                self.pad,
                self.bos,
                self.eos,
                in_origin=output_origin[can_del_word]
                if output_origin is not None
                else None
            )
            output_tokens = _fill_single(
                output_tokens, can_del_word, _tokens, self.pad)
            output_scores = _fill_single(
                output_scores, can_del_word, _scores, 0)
            output_origin = _fill_single(
                output_origin, can_del_word, _origin, 0)
            attn = _fill_single(attn, can_del_word, _attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("del", self.scatter_del(del_pred, can_del_word)))

        # insert placeholders
        can_plh = output_tokens.ne(self.pad).sum(1) < max_lens
        if can_plh.sum() != 0:
            plh_out, _ = self.decoder.forward_plh(
                normalize=True,
                prev_output_tokens=_skip_single(output_tokens, can_plh),
                encoder_out=_skip_encoder_out_single(
                    self.encoder, encoder_out, can_plh),
            )
            if self.use_insert_distribution:
                # plh_pred = torch.round(plh_out.squeeze(-1))
                plh_pred = self.predict_plh_from_out(plh_out, penalty=eos_penalty)
            else:
                if eos_penalty > 0.0:
                    # TODO: adapt to distro-plh
                    plh_out[:, :, 0] = plh_out[:, :, 0] - eos_penalty
                plh_pred = plh_out.max(-1)[1]
            plh_pred = torch.minimum(
                plh_pred,
                torch.tensor(255, device=plh_pred.device,
                             dtype=plh_pred.dtype),
            )

            _tokens, _scores, _origin = _apply_ins_masks(
                output_tokens[can_plh],
                output_scores[can_plh],
                plh_pred,
                self.pad,
                self.unk,
                self.eos,
                in_origin=output_origin[can_plh]
                if output_origin is not None
                else None
            )
            output_tokens = _fill_single(
                output_tokens, can_plh, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_plh, _scores, 0)
            output_origin = _fill_single(output_origin, can_plh, _origin, 0)

            if history is not None:
                history.append(output_tokens.clone())
            if history_ops is not None:
                history_ops.append(
                    ("plh", self.scatter_plh(plh_pred, can_plh)))

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
            _tokens, _scores, _origin = _apply_ins_words(
                output_tokens[can_tok],
                output_scores[can_tok],
                tok_pred,
                tok_score,
                self.unk,
                in_origin=output_origin[can_tok]
                if output_origin is not None
                else None
            )
            output_tokens = _fill_single(
                output_tokens, can_tok, _tokens, self.pad)
            output_scores = _fill_single(output_scores, can_tok, _scores, 0)
            output_origin = _fill_single(output_origin, can_tok, _origin, 0)
            attn = _fill_single(attn, can_tok, tok_attn, 0.0)

            if history is not None:
                history.append(output_tokens.clone())

        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(-1).max()
        output_tokens = output_tokens[:, :cut_off]
        output_scores = output_scores[:, :cut_off]
        if output_origin is not None:
            output_origin = output_origin[:, :cut_off]

        attn = None if attn is None else attn[:, :cut_off, :]

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            output_origin=output_origin,
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

    def initialize_output_tokens(self, encoder_out, multi_src_tokens, retain_origin=False):
        return DecoderOut(
            output_tokens=multi_src_tokens,
            output_scores=torch.zeros_like(
                multi_src_tokens
            ).type_as(encoder_out["encoder_out"][0]),
            output_origin=torch.arange(
                1, multi_src_tokens.size(1) + 1, device=multi_src_tokens.device
            )[None, :, None].expand_as(multi_src_tokens)
            if retain_origin
            else None,
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
        self.Kmax = args.plh_max_num_insert
        self.use_insert_distribution = getattr(args, "continuous_insert", False)
        self.insert_var = getattr(args, "insert_var", 4.0)
        self.clf_num = getattr(args, "clf_num", 1)
        if self.use_insert_distribution:
            self.embed_plh = Embedding(self.clf_num + 2, self.output_embed_dim * 2, None) # predict mean of distribution + weight of 0
        else:
            self.embed_plh = Embedding(self.Kmax, self.output_embed_dim * 2, None)
        self.embed_word_del = Embedding(2, self.output_embed_dim, None)
        self.embed_cmb = nn.Linear(self.output_embed_dim, 1)
        self.embed_tok = nn.Linear(self.output_embed_dim, len(self.dictionary))
        self.num_retrieved = args.num_retrieved
        self.embed_seq_num = Embedding(
            self.num_retrieved + 1, embed_tokens.embedding_dim, None
        )
        self.squash_multi_toks = not getattr(args, "unsquash", False)
        self.cpt_save = 0

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
        mask = multi_toks.ne(self.pad)
        max_length = mask.view(multi_toks.size(0), -1).sum(-1).max()
        seq_index = (multi_toks == self.bos).cumsum(-1)
        squashed_toks = torch.full(
            (multi_toks.size(0), max_length),
            self.pad, dtype=multi_toks.dtype, device=multi_toks.device
        )
        sorted_ = mask.view(multi_toks.size(0), -1).long().sort(descending=True, stable=True, dim=-1)
        flat_multi_toks = multi_toks.view(multi_toks.size(0), -1)
        squashed_toks = flat_multi_toks[
            torch.arange(
                multi_toks.size(0),
                device=multi_toks.device
            )[:, None].expand_as(flat_multi_toks), 
            sorted_[1]
        ][:, :max_length]
        return squashed_toks, seq_index, sorted_[1]

    def multi_unsquash(self, squashed, flat_index, N, L):
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
                    positions_ = self.embed_positions(prev_output_tokens[:, 0, :], trunc_pos=True)
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
                    positions = self.embed_positions(prev_output_tokens[:, 0, :], trunc_pos=True)
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
            # change shape (batch x N x M x p) to (batch x NM x p)
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
                self.embed_positions(prev_output_tokens, trunc_pos=True)
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
                    decoder_padding_mask.size(0),
                    -1
                ),
            )
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
        multi_len = (len(prev_output_tokens.shape) == 3)
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[0],
            layers=self.layers_del,
            multi_len=(len(prev_output_tokens.shape) == 3),
            **unused
        )
        # features: batch x N x M x d
        decoder_out = F.linear(features, self.embed_word_del.weight)

        if "flat_index" in extra:
            decoder_out = self.multi_unsquash(
                decoder_out, extra["flat_index"], extra["N"], extra["L"]
            )
            if extra["attn"] is not None:
                extra["attn"] = self.multi_unsquash(
                    extra["attn"], extra["flat_index"], extra["N"], extra["L"]
                )
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    @ensemble_decoder
    def forward_plh(self, normalize, encoder_out, prev_output_tokens, **unused):
        multi_len = (len(prev_output_tokens.shape) == 3)
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
            features = self.multi_unsquash(
                features, extra["flat_index"], extra["N"], extra["L"]
            )
            if extra["attn"] is not None:
                extra["attn"] = self.multi_unsquash(
                    extra["attn"], extra["flat_index"], extra["N"], extra["L"]
                )
        if multi_len:
            features_cat = torch.cat(
                [features[:, :, :-1, :], features[:, :, 1:, :]],
                -1
            )
        else:
            features_cat = torch.cat(
                [features[:, :-1, :], features[:, 1:, :]],
                -1
            )
        decoder_out = F.linear(features_cat, self.embed_plh.weight)
        if normalize and not self.use_insert_distribution:
            return F.log_softmax(decoder_out, -1), extra["attn"]
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
