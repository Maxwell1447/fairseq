# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import torch
from fairseq import utils
from fairseq.dataclass import ChoiceEnum
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask
from fairseq.utils import new_arange
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguageMultiSourceDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
    FairseqDataset,
    iterators,
)
import logging
import numpy as np

import os
import itertools


logger = logging.getLogger(__name__)

NOISE_CHOICES = ChoiceEnum(["random_delete", "random_mask", "no_noise"])


@dataclass
class TranslationMultiLevenshteinConfig(TranslationConfig):
    noise: NOISE_CHOICES = field(
        default="random_delete", metadata={"help": "type of noise"},
    )
    num_retrieved: int = field(
        default=1,
        metadata={"help": "number of co-edited sequences from the monoling corpus"},
    )
    max_num_retrieved: int = field(
        default=-1,
        metadata={"help": "number of co-edited sequences from the monoling corpus"},
    )
    max_acceptable_retrieved_ratio: float = field(
        default=1.8,
        metadata={
            "help": "Maximum authorized ratio between retrieved examples and target"
        },
    )
    max_positions: int = field(
        default=1024,
        metadata={
            "help": "Maximum size of src/mutli-src sentences"
        },
    )
    load_idf: bool = field(
        default=False,
        metadata={
            "help": "Loads array of token IDF. Used for weighted multi-alignment at training time"
        },
    )


def load_lang_multi_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    num_multi_src,
    max_num_multi_src,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    load_idf=False,
):
    if load_idf and os.path.exists(os.path.join(data_path, f"idf.{src}.bin")):
        idf = torch.from_numpy(np.fromfile(os.path.join(data_path, f"idf.{src}.bin"), dtype=np.float32))
        logger.info(
            "IDF file loaded successfully: {}".format(
                os.path.join(data_path, f"idf.{src}.bin")
            )
        )
    else:
        idf = None
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    multi_src_datasets = []

    for n in range(num_multi_src):
        multi_src_datasets.append([])

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        for n in range(num_multi_src):
            dir1 = split_exists(split_k, src, tgt, tgt + str(n + 1), data_path)
            dir2 = split_exists(split_k, tgt, src, tgt + str(n + 1), data_path)
            if not (dir1 or dir2):
                raise FileNotFoundError(
                    "Retrieval dataset #{} not found: {} ({})".format(
                        n + 1, split, data_path
                    )
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        for n in range(num_multi_src):
            if n >= max_num_multi_src:
                multi_src_datasets[n] = None
                break
            single_src_dataset = data_utils.load_indexed_dataset(
                prefix + tgt + str(n + 1), tgt_dict, dataset_impl
            )
            if single_src_dataset is not None:
                multi_src_datasets[n].append(single_src_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
        for n in range(num_multi_src):
            multi_src_datasets[n] = (
                multi_src_datasets[n][0] 
                if multi_src_datasets[n] is not None and len(multi_src_datasets[n]) > 0 
                else None
            )
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None
        for n in range(num_multi_src):
            if multi_src_datasets[n] is not None:
                multi_src_datasets[n] = ConcatDataset(multi_src_datasets[n], sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
        if multi_src_datasets is not None:
            for n in range(num_multi_src):
                if multi_src_datasets[n] is not None:
                    multi_src_datasets[n] = PrependTokenDataset(
                        multi_src_datasets[n], tgt_dict.bos()
                    )

    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        if multi_src_datasets is not None:
            for n in range(num_multi_src):
                if multi_src_datasets[n] is not None:
                    multi_src_datasets[n] = AppendTokenDataset(
                        multi_src_datasets[n], tgt_dict.index("[{}]".format(tgt))
                    )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    multi_src_sizes = [
        single.sizes if single is not None else None for single in multi_src_datasets
    ]
    return LanguageMultiSourceDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        multi_src_datasets,
        multi_src_sizes,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        idf=idf
    )


@register_task("multi_translation_lev", dataclass=TranslationMultiLevenshteinConfig)
class TranslationMultiLevenshteinTask(TranslationTask):
    """
    Translation (Sequence Generation) task for Levenshtein Transformer
    See `"Levenshtein Transformer" <https://arxiv.org/abs/1905.11006>`_.
    """

    cfg: TranslationMultiLevenshteinConfig
    tokenizer = None

    def max_positions(self):
        return self.cfg.max_positions

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

        self.datasets[split] = load_lang_multi_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            self.cfg.num_retrieved,
            self.cfg.max_num_retrieved 
            if self.cfg.max_num_retrieved > 0 
            else self.cfg.num_retrieved,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            prepend_bos=True,
            load_idf=self.cfg.load_idf
        )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = self.filter_indices_by_size(
                indices,
                dataset,
                max_positions,
                ignore_invalid_inputs,
                max_acceptable_retrieved_ratio=self.cfg.max_acceptable_retrieved_ratio,
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

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.cfg.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.cfg.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.cfg.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError

    def build_generator(self, models, args, **unused):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            beam_size=getattr(args, "decode_with_beam", 1),
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_ratio=getattr(args, "decode_max_ratio", None),
            max_iter=getattr(args, "iter_decode_max_iter", 10),
            retain_origin=getattr(args, "retain_origin", False),
            retain_history=getattr(args, "retain_iter_history", False),
            realigner=getattr(args, "realigner", "no")
        )

    def build_dataset_for_inference(
        self,
        src_tokens,
        src_lengths,
        multi_src_tokens,
        multi_src_sizes,
        constraints=None,
    ):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )
        return LanguageMultiSourceDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            multi_src=multi_src_tokens,
            multi_src_sizes=multi_src_sizes,
            append_bos=True,
        )

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = None
        sample["num_iter"] = update_num

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = None
            sample["num_iter"] = 0
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
