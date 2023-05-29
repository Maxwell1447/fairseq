# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from json import encoder
import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor
# from fairseq import libdual_cuda
import os


@register_criterion("dual_loss")
class DualEncodingLoss(FairseqCriterion):
    def __init__(self, task, label_smoothing, bag_of_word):
        super().__init__(task)
        self.label_smoothing = label_smoothing
        self.bag_of_word = bag_of_word

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.1,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def _compute_loss(
        self, encoding_x, encoding_y
    ):
        """
        encoding_x: batch x d_model
        encoding_y: batch x d_model
        """
        s_xy = torch.matmul(encoding_x, encoding_y.T)

        # loss = - s_xy.diag()
        # norm = torch.logsumexp(s_xy, dim=-1)
        # loss = (loss + norm).sum()
        loss = -torch.nn.functional.log_softmax(s_xy, dim=1).diag().mean()

        if self.label_smoothing > 0.0:
            # loss_ls = -(s_xy.sum() / s_xy.size(0) - norm.sum())
            loss_ls = -torch.nn.functional.log_softmax(s_xy, dim=1).mean(-1).mean()
            # with torch.no_grad():
            #     print("other ls = ", -torch.nn.functional.log_softmax(s_xy, dim=1).mean(-1).sum())
            # print(
            #     "\n\n\n---------loss alignment--------",
            #     "\nloss=", loss.detach().cpu().item(),
            #     "\nloss_ls=", loss_ls.detach().cpu().item(),
            #     "\ns_xy=", s_xy.detach().sum().cpu().item(),
            #     "\nbsz=", s_xy.size(0)
            # )
            loss = (1 - self.label_smoothing) * loss + self.label_smoothing * loss_ls
        # else:
        #     print("no label smoothing")
            
        return loss

    def _compute_bag_of_word_loss(self, logits, tokens, bos=0, eos=2, pad=1, factor=1.):
        """
        logits: batch x voc_size
        tokens: batch x length
        """
        # batch x voc_size
        bow_mask = libdual_cuda.get_bow_mask_from_sequence(tokens, logits.size(1), bos, eos, pad)

        loss = -torch.nn.functional.log_softmax(logits, dim=1)[bow_mask.bool()]

        # norm = torch.logsumexp(logits, dim=-1)

        # # print(logits.shape, bow_mask.shape, norm.shape)
        # print("bow size", bow_mask.sum(-1).float().mean().data.item())

        # loss = -(logits * bow_mask).sum(-1) + norm * bow_mask.sum(-1)

        return loss.mean() * factor

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        # B x L
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, tgt_lengths = (
            sample["net_input"]["tgt_tokens"],
            sample["net_input"]["tgt_lengths"],
        )
        # print("src", src_tokens.shape)
        # print("tgt", tgt_tokens.shape)
        outputs = model(src_tokens, src_lengths, tgt_tokens, tgt_lengths)
        loss_align = self._compute_loss(
            outputs["encoder_src_out"],
            outputs["encoder_tgt_out"],
        )
        # print(">>> loss   = ", loss_align.data.item())
        sample_size = 1
        logging_output = dict()
        

        if self.bag_of_word:
            assert "src_bow_logits" in outputs
            assert "tgt_bow_logits" in outputs
            loss_bow_src = self._compute_bag_of_word_loss(
                outputs["src_bow_logits"],
                tgt_tokens,
            )
            loss_bow_tgt = self._compute_bag_of_word_loss(
                outputs["tgt_bow_logits"],
                src_tokens,
            )
            logging_output["bow_src_loss"] = loss_bow_src.data
            logging_output["bow_tgt_loss"] = loss_bow_tgt.data
            # print("+++ losses = ", loss_bow_src.data.item(), loss_bow_tgt.data.item())
            loss = loss_align + loss_bow_src + loss_bow_tgt
        else:
            loss = loss_align

        logging_output.update({
            "loss": loss.data,
            "nll_loss": loss.data,
            "align_loss": loss_align.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        })
        
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        nll_loss = utils.item(sum(log.get("nll_loss", 0) for log in logging_outputs))
        align_loss = utils.item(sum(log.get("align_loss", 0) for log in logging_outputs))
        # print("logged loss", loss)
        # print("logged nll_loss", loss)
        # print("logged align_loss", align_loss)
        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "align_loss", align_loss / sample_size / math.log(2), sample_size, round=3
        )
        for log_ in logging_outputs:
            if "bow_src_loss" in log_:
                bow_src_loss = utils.item(sum(log["bow_src_loss"]for log in logging_outputs))
                metrics.log_scalar(
                    "bow_src_loss",
                    bow_src_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            if "bow_tgt_loss" in log_:
                bow_tgt_loss = utils.item(sum(log["bow_tgt_loss"]for log in logging_outputs))
                metrics.log_scalar(
                    "bow_tgt_loss",
                    bow_tgt_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
                    sample_size,
                    round=3,
                )
            break

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
