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
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def _compute_loss(
        self, encoding_x, encoding_y, label_smoothing=0.0
    ):
        """
        encoding_x: batch x d_model
        encoding_y: batch x d_model
        """
        s_xy = torch.matmul(encoding_x, encoding_y.T)

        loss = - s_xy.diag()
        norm = torch.logsumexp(s_xy, dim=-1)
        loss = (loss + norm).sum()
        if label_smoothing > 0.0:
            loss_ls = s_xy.sum() / s_xy.size(0) - norm.sum()
            loss = label_smoothing * loss + (1 - label_smoothing) * loss_ls
        return loss / encoding_x.size(0)

    def _compute_bag_of_word_loss(self, logits, tokens, lengths, label_smoothing=0.0):
        """
        encoding_x: batch x d_model
        encoding_y: batch x d_model
        """
        bow_mask = torch.arange(tokens.size(1))[None, :].expand_as(tokens) < lengths
        tokens[~bow_mask]
        norm = torch.logsumexp(logits, dim=-1)

        return logits.new_zeros(1)[0]

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

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
        
        outputs = model(src_tokens, src_lengths, tgt_tokens, tgt_lengths)
        loss = self._compute_loss(
            outputs["encoder_src_out"],
            outputs["encoder_tgt_out"],
        )

        if self.bag_of_word:
            
            loss_bow_src = self._compute_bag_of_word_loss(
                outputs["src_bow_logits"],
                src_tokens,
                src_lengths
            )
            loss_bow_tgt = self._compute_bag_of_word_loss(
                outputs["tgt_bow_logits"],
                tgt_tokens,
                tgt_lengths
            )
            logging_output["bow_src_loss"] = loss_bow_src.data
            logging_output["bow_tgt_loss"] = loss_bow_tgt.data
            loss += loss_bow_src + loss_bow_tgt

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data,
            "nll_loss": loss.data,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        sample_size = utils.item(
            sum(log.get("sample_size", 0) for log in logging_outputs)
        )
        loss = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        metrics.log_scalar(
            "alignment_loss", loss / sample_size / math.log(2), sample_size, round=3
        )

        if "bag_of_word_loss" in logging_outputs:
            bag_of_word_loss = utils.item(sum(log["bag_of_word_loss"]for log in logging_outputs))
            metrics.log_scalar(
                "bag_of_word_loss",
                bag_of_word_loss / sample_size / math.log(2) if sample_size > 0 else 0.0,
                sample_size,
                round=3,
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
