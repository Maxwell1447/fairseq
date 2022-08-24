# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from fairseq.utils import new_arange



@dataclass
class DualEncodingConfig(TranslationConfig):
    ...

@register_task("dual_encoding", dataclass=DualEncodingConfig)
class DualEncodingTask(TranslationTask):
    """
    """

    cfg: DualEncodingConfig

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
        )

    def build_generator(self, models, args, **unused):
        raise NotImplementedError("generator of embedding not implemented yet")
        # add models input to match the API for SequenceGenerator
        # from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        # return IterativeRefinementGenerator(
        #     self.target_dictionary,
        #     eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
        #     max_iter=getattr(args, "iter_decode_max_iter", 10),
        #     beam_size=getattr(args, "iter_decode_with_beam", 1),
        #     reranking=getattr(args, "iter_decode_with_external_reranker", False),
        #     decoding_format=getattr(args, "decoding_format", None),
        #     adaptive=not getattr(args, "iter_decode_force_max_iter", False),
        #     retain_history=getattr(args, "retain_iter_history", False),
        # )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
