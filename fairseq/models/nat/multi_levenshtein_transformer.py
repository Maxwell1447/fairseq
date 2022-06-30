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
        self.Kmax = args.plh_max_num_insert
        self.full_levt = args.basic_levt_align = getattr(args, "basic_levt_align", False)
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
            "--plh-max-num-insert",
            default=64,
            type=int,
            help="Number of placeholders that can be added between 2 consecutive tokens",
        )
        parser.add_argument(
            "--num-retrieved",
            default=1,
            type=int,
            help="Number of sentences retrieved, then edited together to form the final sentence",
        )
        parser.add_argument(
            "--basic-levt-align",
            action="store_true",
            help="Use the lev distance to individually align to-be-edited sequences.",
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = MultiLevenshteinTransformerDecoder(
            args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            decoder.apply(init_bert_params)
        return decoder

    def regularize_shapes(self, x, ys, y):
        bsz = x.size(0)
        M = max(x.size(-1), ys.size(-1), y.size(-1))
        N = ys.size(1)
        shape = (bsz, N + 2, M)
        X = x.new(*shape).fill_(self.pad)
        X[:, 0, : x.size(-1)] = x
        X[:, -1, : y.size(-1)] = y
        X[:, 1:-1, : ys.size(-1)] = ys

        return X[:, 0, :], X[:, 1:-1, :], X[:, -1, :]

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

    def forward(self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs):
        src_tokens, prev_output_tokens, tgt_tokens = self.regularize_shapes(
            src_tokens, prev_output_tokens, tgt_tokens
        )
        if self.full_levt:
            mask_good = torch.zeros(src_tokens.size(0), device=src_tokens.device, dtype=torch.bool)
        else:
            mask_good = torch.ones(src_tokens.size(0), device=src_tokens.device, dtype=torch.bool)
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
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        
        del_tgt = None
        del_mask = None
        # choose init
        mask_star = self.get_mask_from_prob(prev_output_tokens.size(0), self.beta) # for pi del
        with torch.no_grad():
            res_star = pi_star(
                prev_output_tokens[mask_star][mask_good[mask_star]],
                tgt_tokens[mask_star][mask_good[mask_star]],
                pad_symbol=self.pad,
                plh_symbol=self.unk,
                Kmax=self.Kmax,
                device=src_tokens.device,
            )
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
                    ############################ prev del levt
                    del_tgt_bad.append(_get_del_targets(
                        prev_output_tokens[mask_star][~mask_good[mask_star]][:, n], 
                        tgt_tokens[mask_star][~mask_good[mask_star]], 
                        self.pad
                    ).unsqueeze(1))
                    del_mask_bad.append(prev_output_tokens[mask_star][~mask_good[mask_star]][:, n].ne(self.pad).unsqueeze(1))
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
                    ############################ ins levt
                    masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
                        y_plh_bad_, tgt_tokens[mask_star][~mask_good[mask_star]], self.pad, self.unk
                    )
                    y_cmb_bad.append(masked_tgt_tokens)
                    # if tok_mask_bad is None:
                    #     tok_mask_bad = masked_tgt_masks
                    # else:
                    #     tok_mask_bad = (tok_mask_bad & masked_tgt_masks)
                    plh_tgt_bad.append(mask_ins_targets.clamp(min=0, max=63).unsqueeze(1))  # for safe prediction
                    plh_mask_bad.append(y_plh_bad_[:, 1:].ne(self.pad).unsqueeze(1))
                del_tgt_bad = torch.cat(del_tgt_bad, dim=1)
                del_mask_bad = torch.cat(del_mask_bad, dim=1)
                y_plh_bad = torch.cat([t.unsqueeze(1) for t in y_plh_bad], dim=1)
                y_cmb_bad = torch.cat([t.unsqueeze(1) for t in y_cmb_bad], dim=1)
                plh_tgt_bad = torch.cat(plh_tgt_bad, dim=1)
                plh_mask_bad = torch.cat(plh_mask_bad, dim=1)
                cmb_mask_bad = y_cmb_bad.ne(self.pad)
                cmb_tgt_bad = (cmb_mask_bad & y_cmb_bad.ne(self.unk))
                y_tok_bad = tgt_tokens[mask_star][~mask_good[mask_star]].clone()
                y_tok_bad[(cmb_tgt_bad == self.unk).all(1)] = self.unk
                tok_mask_bad = y_tok_bad.ne(self.unk)
                cmb_tgt_bad = cmb_tgt_bad.long() # TRANSPOSE ??????????????

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
                device=src_tokens.device
            )
            cmb_tgt = handle_all_plh_case(cmb_tgt, y_tok, y_cmb, self.unk)

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
        tok_out, _ = self.decoder.forward_tok(
            normalize=False, prev_output_tokens=y_tok, encoder_out=encoder_out,
        )
        cmb_out = cmb_out.transpose(1, 2)

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

        # ############################ pred for del
        # if self.decoder.sampling_for_deletion:
        #     word_predictions = torch.multinomial(
        #         F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1
        #     ).view(word_ins_out.size(0), -1)
        # else:
        #     word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        # word_predictions.masked_scatter_(
        #     ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        # )
        # ############################ del correct levt
        # word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        # word_del_out, _ = self.decoder.forward_word_del(
        #     normalize=False,
        #     prev_output_tokens=word_predictions,
        #     encoder_out=encoder_out,
        # )
        # word_del_masks = word_predictions.ne(self.pad)

        output = dict()
        
        output["prev_word_del"] = {
            "out": del_out,
            "tgt": del_tgt,
            "mask": del_mask,
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

        return output      


    # def forward_debug(
    #     self, x, y_init_star, src_lengths=None, **kwargs
    # ):
    #     with torch.no_grad():
    #         # encoding
    #         encoder_out = self.encoder(x, src_lengths=src_lengths, **kwargs)
    #         # print("encoder out", encoder_out)

    #         # choose init
    #         y_del = y_init_star

    #         del_out, _ = self.decoder.forward_del(
    #             normalize=True,
    #             prev_output_tokens=y_del,
    #             encoder_out=encoder_out,
    #         )

    #         # plh_out, _ = self.decoder.forward_plh(
    #         #     normalize=True, prev_output_tokens=y_plh, encoder_out=encoder_out,
    #         # )

    #         # cmb_out, _ = self.decoder.forward_cmb(
    #         #     normalize=True, prev_output_tokens=y_cmb, encoder_out=encoder_out,
    #         # )

    #         # tok_out, _ = self.decoder.forward_tok(
    #         #     normalize=True, prev_output_tokens=y_tok, encoder_out=encoder_out,
    #         # )

    #     return {
    #         "del_out": del_out,
    #         # "plh_out": plh_out,
    #         # "cmb_out": cmb_out,
    #         # "tok_out": tok_out,
    #     }

    def forward_decoder(
        self, decoder_out, encoder_out, eos_penalty=0.0, max_ratio=None, **kwargs
    ):

        output_tokens = decoder_out.output_tokens  # B x N x M
        history = decoder_out.history

        max_lens = 255
        verbose = 1

        # delete words
        # do not delete tokens if it is <s> </s>

        can_del_word = (output_tokens.ne(self.pad).sum(-1) > 2).any(-1)

        if can_del_word.sum() != 0:  # we cannot delete, skip

            # Skip ignores batch element with no possible deletion
            del_out, _ = self.decoder.forward_del(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_del_word),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_del_word),
            )
            del_out = F.softmax(del_out, -1)
            del_pred = del_out.max(-1)[1].bool()
            if verbose:
                ranger = (
                    torch.arange(2, dtype=del_out.dtype, device=del_out.device)
                    .view(1, 1, 1, -1)
                    .expand(del_out.shape)
                )
                res = (ranger * del_out).sum(-1).mean(0)
                #                print("del_pred mean", del_pred.float().mean(0))
                print("del_pred expect", res)

            _tokens = apply_del(
                output_tokens[can_del_word], del_pred, self.pad, self.bos, self.eos,
            )
            output_tokens = _fill(
                output_tokens, can_del_word, _tokens, self.pad)

            if history is not None:
                history.append(output_tokens.clone())

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
            plh_out = F.softmax(plh_out, -1)
            plh_pred = plh_out.max(-1)[1]
            plh_pred = torch.minimum(
                plh_pred,
                torch.tensor(255, device=plh_pred.device,
                             dtype=plh_pred.dtype),
            )
            if verbose:
                ranger = (
                    torch.arange(
                        plh_out.size(-1), dtype=plh_out.dtype, device=plh_out.device
                    )
                    .view(1, 1, 1, -1)
                    .expand(plh_out.shape)
                )
                res = (ranger * plh_out).sum(-1).mean(0)
                print("plh_pred expect", res)
            #                print("plh_pred mean", plh_pred.float().mean(0))

            _tokens = apply_plh(
                output_tokens[can_plh], plh_pred, self.pad, self.unk, self.eos,
            )
            output_tokens = _fill(output_tokens, can_plh, _tokens, self.pad)

            if history is not None:
                history.append(output_tokens.clone())
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
        ] = self.unk # makes sense I guess

        # merge sequences
        cmb_out, _ = self.decoder.forward_cmb(
            normalize=True, prev_output_tokens=output_tokens, encoder_out=encoder_out,
        )
        plh_out = F.softmax(plh_out, -1)
        cmb_pred = cmb_out[:, :, :, 1]
        #        cmb_pred = cmb_out.max(-1)[1]
        if verbose:
            ranger = (
                torch.arange(
                    cmb_out.size(-1), dtype=cmb_out.dtype, device=cmb_out.device
                )
                .view(1, 1, 1, -1)
                .expand(cmb_out.shape)
            )
            res = (ranger * cmb_out).sum(-1).mean(0)
            print("cmb_pred expect", res)
        #            print("cmb_pred mean", cmb_out.max(-1)[1].float().mean(0))

        output_tokens = apply_cmb(
            output_tokens, cmb_pred, self.pad, self.bos, self.eos, self.unk,
        )
        if history is not None:
            history.append(output_tokens.clone())

        # insert tok
        can_tok = output_tokens.eq(self.unk).sum(1) > 0
        if can_tok.sum() != 0:
            # print("before tok >>>", _skip(output_tokens, can_tok).cpu().numpy().tolist())
            tok_out, _ = self.decoder.forward_tok(
                normalize=True,
                prev_output_tokens=_skip(output_tokens, can_tok),
                encoder_out=_skip_encoder_out(
                    self.encoder, encoder_out, can_tok),
            )
            tok_out[:, 3] = tok_out[:, 0] - 10
            _, tok_pred = tok_out.max(-1)
            tok_out = F.softmax(tok_out, -1)
            # if verbose:
            #     ranger = (
            #         torch.arange(
            #             tok_out.size(-1), dtype=tok_out.dtype, device=tok_out.device
            #         )
            #         .view(1, 1, 1, -1)
            #         .expand(tok_out.shape)
            #     )
            #     res = (ranger * tok_pred).sum(-1).mean(0)
            #     print("tok_pred expect", res)
            #                print("tok_pred mean", tok_pred.float().mean(0))

            _tokens = apply_tok(output_tokens[can_tok], tok_pred, self.unk,)

            output_tokens = _fill(output_tokens, can_tok, _tokens, self.pad)

            # print("after tok >>>", _skip(output_tokens, can_tok).cpu().numpy().tolist())

            if history is not None:
                history.append(output_tokens.clone())
        else:
            print("=== no <unk> to fill????")
        # delete some unnecessary paddings
        cut_off = output_tokens.ne(self.pad).sum(-1).max()
        output_tokens = output_tokens[:, :cut_off]
        # output_score = decoder_out.output_scores[:, :cut_off]
        output_scores = torch.zeros_like(
            output_tokens, dtype=torch.float, device=output_tokens.device)

        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            attn=None if decoder_out.attn is None else decoder_out.attn[:, :cut_off, :],
            history=history,
        )

    def initialize_output_tokens(self, encoder_out, multi_src_tokens):

        return DecoderOut(
            output_tokens=multi_src_tokens,
            output_scores=torch.zeros(
                multi_src_tokens.size(0), multi_src_tokens.size(2)
            ).type_as(encoder_out["encoder_out"][0]),
            attn=None,
            step=0,
            max_step=0,
            history=None,
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
        # prev_output_tokens: batch x N x M
        if multi_len and len(prev_output_tokens.shape) > 1:
            if self.embed_positions is not None:
                positions = self.embed_positions(prev_output_tokens[:, 0, :])
                positions = positions.unsqueeze(1).expand(
                    (prev_output_tokens.size(0),
                     prev_output_tokens.size(1),
                     prev_output_tokens.size(2),
                     positions.size(-1),)
                )
            else:
                positions = None
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

            # print("tok_emb", tok_emb)
            # print("seq_emb", seq_emb)
            # print("positions", positions)

            # change shape (batch x N x M x p) to (batch x NM x p)
            positions = positions.view(
                prev_output_tokens.size(0),
                prev_output_tokens.size(1) * prev_output_tokens.size(2),
                self.embed_seq_num.embedding_dim,
            )
            seq_emb = seq_emb.view(
                prev_output_tokens.size(0),
                prev_output_tokens.size(1) * prev_output_tokens.size(2),
                self.embed_seq_num.embedding_dim,
            )
            tok_emb = tok_emb.view(
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
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        if multi_len:
            shape = (
                prev_output_tokens.size(0),
                prev_output_tokens.size(1),
                prev_output_tokens.size(2),
                -1,
            )
            x = x.view(shape)
            if attn is not None:
                attn = attn.view(shape)

        return x, {"attn": attn, "inner_states": inner_states}

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
        # features: batch x N x M x d
        # print("features", features.shape)
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
        # print("features_cat", features_cat.shape)
        # print("self.embed_plh.weight", self.embed_plh.weight.shape)
        decoder_out = F.linear(features_cat, self.embed_plh.weight)
        if normalize:
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
        # features: batch x N x M x d
        decoder_out = (
            self.embed_cmb(features).transpose(1, 2).squeeze(-1)
        )  # batch x M x N
        decoder_out = torch.sigmoid(decoder_out)
        decoder_out = torch.stack(
            (decoder_out, 1 - decoder_out), dim=-1
        )  # batch x M x N x 2

        # decoder_out: batch x M x N

        return decoder_out[:, :, :, :], extra["attn"]

    @ensemble_decoder
    def forward_tok(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, extra = self.extract_features_multi(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.early_exit[2],
            layers=self.layers,
            **unused
        )
        # features: batch x M x d
        decoder_out = self.output_layer(features)
        if normalize:
            return F.log_softmax(decoder_out, -1), extra["attn"]
        return decoder_out, extra["attn"]

    # def extract_features(
    #     self,
    #     prev_output_tokens,
    #     encoder_out=None,
    #     early_exit=None,
    #     layers=None,
    #     **unused
    # ):
    #     """
    #     Similar to *forward* but only return features.
    #     Inputs:
    #         prev_output_tokens: Tensor(B, T)
    #         encoder_out: a dictionary of hidden states and masks

    #     Returns:
    #         tuple:
    #             - the decoder's features of shape `(batch, tgt_len, embed_dim)`
    #             - a dictionary with any model-specific outputs
    #         the LevenshteinTransformer decoder has full-attention to all generated tokens
    #     """
    #     # embed positions
    #     positions = (
    #         self.embed_positions(prev_output_tokens)
    #         if self.embed_positions is not None
    #         else None
    #     )

    #     # embed tokens and positions
    #     x = self.embed_scale * self.embed_tokens(prev_output_tokens)
    #     if self.project_in_dim is not None:
    #         x = self.project_in_dim(x)

    #     if positions is not None:
    #         x += positions
    #     x = self.dropout_module(x)

    #     # B x T x C -> T x B x C
    #     x = x.transpose(0, 1)
    #     attn = None
    #     inner_states = [x]

    #     # decoder layers
    #     decoder_padding_mask = prev_output_tokens.eq(self.padding_idx)
    #     layers = self.layers if layers is None else layers
    #     early_exit = len(layers) if early_exit is None else early_exit
    #     for _, layer in enumerate(layers[:early_exit]):
    #         x, attn, _ = layer(
    #             x,
    #             encoder_out["encoder_out"][0]
    #             if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
    #             else None,
    #             encoder_out["encoder_padding_mask"][0]
    #             if (
    #                 encoder_out is not None
    #                 and len(encoder_out["encoder_padding_mask"]) > 0
    #             )
    #             else None,
    #             self_attn_mask=None,
    #             self_attn_padding_mask=decoder_padding_mask,
    #         )
    #         inner_states.append(x)

    #     if self.layer_norm:
    #         x = self.layer_norm(x)

    #     # T x B x C -> B x T x C
    #     x = x.transpose(0, 1)

    #     if self.project_out_dim is not None:
    #         x = self.project_out_dim(x)

    #     return x, {"attn": attn, "inner_states": inner_states}

    # @ensemble_decoder
    # def forward_mask_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
    #     features, extra = self.extract_features(
    #         prev_output_tokens,
    #         encoder_out=encoder_out,
    #         early_exit=self.early_exit[1],
    #         layers=self.layers_msk,
    #         **unused
    #     )
    #     features_cat = torch.cat([features[:, :-1, :], features[:, 1:, :]], 2)
    #     decoder_out = F.linear(features_cat, self.embed_mask_ins.weight)
    #     if normalize:
    #         return F.log_softmax(decoder_out, -1), extra["attn"]
    #     return decoder_out, extra["attn"]

    # @ensemble_decoder
    # def forward_word_ins(self, normalize, encoder_out, prev_output_tokens, **unused):
    #     features, extra = self.extract_features(
    #         prev_output_tokens,
    #         encoder_out=encoder_out,
    #         early_exit=self.early_exit[2],
    #         layers=self.layers,
    #         **unused
    #     )
    #     decoder_out = self.output_layer(features)
    #     if normalize:
    #         return F.log_softmax(decoder_out, -1), extra["attn"]
    #     return decoder_out, extra["attn"]

    # @ensemble_decoder
    # def forward_word_del(self, normalize, encoder_out, prev_output_tokens, **unused):
    #     features, extra = self.extract_features(
    #         prev_output_tokens,
    #         encoder_out=encoder_out,
    #         early_exit=self.early_exit[0],
    #         layers=self.layers_del,
    #         **unused
    #     )
    #     decoder_out = F.linear(features, self.embed_word_del.weight)
    #     if normalize:
    #         return F.log_softmax(decoder_out, -1), extra["attn"]
    #     return decoder_out, extra["attn"]

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
