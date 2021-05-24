from __future__ import division

import math
import torch
import torch.utils.data
from collections import defaultdict
import onmt
from onmt.speech.Augmenter import Augmenter
from onmt.modules.dropout import switchout
import numpy as np
from .batch_utils import allocate_batch

"""
Data management for sequence-to-sequence models
Two basic classes: 
- Batch stores the input / output sequences, grouped into tensors with the same length (by padding)
- Dataset stores all of the data and 
"""


def merge_data(data, align_right=False, type='text', augmenter=None, upsampling=False, feature_size=40):
    """
            Assembling the individual sequences into one single tensor, included padding
            :param feature_size:
            :param upsampling:
            :param data: the list of sequences
            :param align_right: aligning the sequences w.r.t padding
            :param type: text or audio
            :param augmenter: for augmentation in audio models
            :return:
            """
    # initialize with batch_size * length
    # TODO: rewrite this function in Cython
    if type == "text":
        lengths = [x.size(0) for x in data]
        positions = [torch.arange(length_) for length_ in lengths]
        max_length = max(lengths)
        tensor = data[0].new(len(data), max_length).fill_(onmt.constants.PAD)
        pos = tensor.new(*tensor.size()).fill_(0)

        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            tensor[i].narrow(0, offset, data_length).copy_(data[i])
            pos[i].narrow(0, offset, data_length).copy_(positions[i])

        return tensor, pos, lengths

    elif type == "audio":

        # First step: on-the-fly processing for the samples
        # Reshaping: either downsampling or upsampling
        # On the fly augmentation
        samples = []

        for i in range(len(data)):
            sample = data[i]

            if augmenter is not None:
                sample = augmenter.augment(sample)

            if upsampling:
                sample = sample.view(-1, feature_size)

            samples.append(sample)

        # compute the lengths afte on-the-fly processing
        lengths = [x.size(0) for x in samples]

        max_length = max(lengths)

        # allocate data for the batch speech
        feature_size = samples[0].size(1)
        batch_size = len(data)

        # feature size + 1 because the last dimension is created for padding
        tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(onmt.constants.PAD)

        for i in range(len(samples)):
            sample = samples[i]

            data_length = sample.size(0)
            offset = max_length - data_length if align_right else 0

            tensor[i].narrow(0, offset, data_length).narrow(1, 1, sample.size(1)).copy_(sample)
            # in padding dimension: 0 is not padded, 1 is padded
            tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

        return tensor, None, lengths
    else:
        raise NotImplementedError


def collect_fn(src_data, tgt_data,
               src_lang_data, tgt_lang_data,
               src_align_right, tgt_align_right,
               src_type='text',
               augmenter=None, upsampling=False,
               bilingual=False, token_level_lang=False, vocab_mask=None, bidirectional=False, multidataset=False, en_id=None):

    tensors = dict()
    if src_data is not None:
        source_full, source_pos, src_lengths = merge_data(src_data, align_right=src_align_right,
                                                                               type=src_type, augmenter=augmenter,
                                                                               upsampling=upsampling, feature_size=40)
        source_full = source_full.transpose(0, 1).contiguous()  # B, T --> T, B

        tensors['src_type'] = src_type

        if not bidirectional:  # as normal
            tensors['source'] = source_full
            if source_pos is not None:
                tensors['source_pos'] = source_pos.transpose(0, 1)
            tensors['src_lengths'] = torch.LongTensor(src_lengths)
            src_lengths = tensors['src_lengths']
            tensors['src_size'] = sum(src_lengths)
            # tensors['src_lengths'] = src_lengths

        else:   # source sentences have BOS and EOS appended
            tensors['source'] = source_full[1:-1]   # strip BOS and EOS to form input
            tensors['source_input'] = source_full[:-1]
            tensors['source_output'] = source_full[1:]

            if source_pos is not None:
                tensors['source_pos'] = source_pos.t().contiguous()[:-1]

            tensors['src_size'] = sum([len(x) - 2 for x in src_data])
            tensors['src_lengths'] = torch.LongTensor([i - 2 for i in src_lengths])  # -2 cuz BOS and EOS
            src_lengths = tensors['src_lengths']

    if tgt_data is not None:
        target_full, target_pos, tgt_lengths = merge_data(tgt_data, align_right=tgt_align_right)
        target_full = target_full.t().contiguous()  # transpose BxT to TxB

        tensors['target'] = target_full

        if bidirectional:   # strip BOS and EOS to form input
            tensors['target_as_source'] = target_full[1:-1]

        tensors['target_input'] = target_full[:-1]
        tensors['target_output'] = target_full[1:]
        if target_pos is not None:
            tensors['target_pos'] = target_pos.t().contiguous()[:-1]
        tgt_size = sum([len(x) - 1 for x in tgt_data])
        tensors['tgt_lengths'] = torch.LongTensor([i - 2 for i in tgt_lengths])
        tgt_lengths = tensors['tgt_lengths']
    else:
        tgt_size = 0
        tensors['tgt_lengths'] = None

    tensors['tgt_size'] = tgt_size
    tensors['size'] = len(src_data) if src_data is not None else len(tgt_data)

    if src_lang_data is not None:
        tensors['source_lang'] = torch.cat(src_lang_data).long()
        if token_level_lang:  # prepare token-level language prediction targets
            if multidataset:
                # In case of multi dataset, there's only one language per batch
                tensors['source_lang'] = tensors['source_lang'].repeat(tensors['source'].shape[1])
            sl_ = tensors['source_lang']  # sl_.shape[0]
            out_dims = (max(src_lengths).item(), sl_.shape[0])  # T, B
            out_tensor = sl_.data.new(*out_dims).fill_(onmt.constants.PAD)
            for i, v in enumerate(sl_):
                # lan ID starts with 0, but pred label values should start with 1 (due to padding)
                out_tensor[:(src_lengths[i]), i] = v + 1
            # print('PREDICTION SHAPE, before', out_tensor)

            # Convert labels to en vs non-en
            if en_id:
                out_tensor[out_tensor == en_id] = 1 # English between 1
                out_tensor[torch.logical_and(out_tensor != 1, out_tensor != onmt.constants.PAD)] = 2
            tensors['targets_source_lang'] = out_tensor

    if tgt_lang_data is not None:
        tensors['target_lang'] = torch.cat(tgt_lang_data).long()

        if token_level_lang and bidirectional:  # prepare token-level language prediction targets, only do this under bidirectional translation
            if multidataset:
                # In case of multi dataset, there's only one language per batch
                tensors['target_lang'] = tensors['target_lang'].repeat(tensors['target'].shape[1])
            sl_ = tensors['target_lang']  # sl_.shape[0]
            # out_dims = (max(tensors['tgt_lengths']).item(), sl_.shape[0])  # T, B
            out_dims = (max(tgt_lengths), sl_.shape[0])
            out_tensor = sl_.data.new(*out_dims).fill_(onmt.constants.PAD)
            for i, v in enumerate(sl_):
                # lan ID starts with 0, but pred label values should start with 1 (due to padding)
                out_tensor[:(src_lengths[i]), i] = v + 1

            tensors['targets_target_lang'] = out_tensor

    tensors['vocab_mask'] = vocab_mask

    return LightBatch(tensors)


def rewrap(light_batch):
    """
    Currently this light batch is used in data collection to avoid pickling error
    After that it is converted to Batch
    :param light_batch:
    :return:
    """
    return Batch(light_batch.tensors)


class Batch(object):
    # An object to manage the data within a minibatch
    def __init__(self, tensors):
        self.tensors = defaultdict(lambda: None, tensors)
        self.src_size = tensors['src_size']
        self.tgt_size = tensors['tgt_size']
        self.size = tensors['size']
        self.src_lengths = tensors['src_lengths']
        self.tgt_lengths = tensors['tgt_lengths']
        self.has_target = True if self.tensors['target'] is not None else False
        self.vocab_mask = tensors['vocab_mask']

        self.source = tensors['source']
        self.source_pos = tensors['source_pos']
        self.targets_source_lang = tensors['targets_source_lang'] if 'targets_source_lang' in tensors else None

    def get(self, name):
        if name in self.tensors:
            return self.tensors[name]
        else:
            return None

    def cuda(self, fp16=False, device=None):
        """
        Send the minibatch data into GPU. Old-fashioned without the 'device' control
        :param device: default = None (default CUDA device)
        :param fp16:
        :return: None
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    if isinstance(k, torch.Tensor):
                        v = tensor[k]
                        tensor[k] = v.cuda(device=device)
            elif tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    if tensor.type() == "torch.FloatTensor" and fp16:
                        self.tensors[key] = tensor.half()
                    self.tensors[key] = self.tensors[key].cuda(device=device)
            else:
                continue

    # def __init__(self, src_data, tgt_data=None,
    #              src_lang_data=None, tgt_lang_data=None,
    #              src_type='text',
    #              src_align_right=False, tgt_align_right=False,
    #              augmenter=None, upsampling=False,
    #              merge=False, token_level_lang=False, **kwargs):
    #     """
    #     :param src_data: list of source tensors
    #     :param tgt_data: list of target tensors
    #     :param src_atb_data: list of attributes/features for the source (TB finished)
    #     :param tgt_atb_data: list of attributes/features for the target (TB finished)
    #     :param src_type: text or audio
    #     :param src_align_right: if the source sequences are aligned to the right
    #     :param tgt_align_right: if the target sequences are aligned to the right
    #     (default False and maybe never changed unless new models need)
    #     :param reshape_speech: the number of frames to be reshaped
    #     :param augmenter: using augmentation for speech
    #     :param merge: if the two sequences are going to be merged for Relative Transformer
    #     """
    #
    #     self.tensors = defaultdict(lambda: None)
    #     self.has_target = False
    #     self.src_type = src_type
    #     # self.upsampling = kwargs.get('upsampling', False)
    #     self.upsampling = upsampling
    #     self.feature_size = kwargs.get('feature_size', 40)
    #     # self.reshape_speech = reshape_speech
    #     self.src_align_right = src_align_right
    #     if merge:
    #         self.src_align_right = True
    #
    #     self.tgt_align_right = tgt_align_right
    #
    #     if src_data is not None:
    #         self.tensors['source'], self.tensors['source_pos'], self.src_lengths = \
    #                                                                 self.collate(src_data,
    #                                                                              align_right=self.src_align_right,
    #                                                                              type=self.src_type,
    #                                                                              augmenter=augmenter)
    #         self.tensors['source'] = self.tensors['source'].transpose(0, 1).contiguous()  # B, T --> T, B
    #         if self.tensors['source_pos'] is not None:
    #             self.tensors['source_pos'] = self.tensors['source_pos'].transpose(0, 1)
    #         self.tensors['src_length'] = torch.LongTensor(self.src_lengths)
    #         self.src_lengths = self.tensors['src_length']
    #         self.src_size = sum(self.src_lengths)
    #     else:
    #         self.src_size = 0
    #
    #     if tgt_data is not None:
    #         target_full, target_pos, self.tgt_lengths = self.collate(tgt_data, align_right=self.tgt_align_right)
    #         target_full = target_full.t().contiguous()  # transpose BxT to TxB
    #         self.tensors['target'] = target_full
    #         self.tensors['target_input'] = target_full[:-1]
    #         self.tensors['target_output'] = target_full[1:]
    #         self.tensors['target_pos'] = target_pos.t().contiguous()[:-1]
    #         self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)
    #         self.has_target = True
    #         self.tgt_size = sum([len(x) - 1 for x in tgt_data])
    #
    #     else:
    #         self.tgt_size = 0
    #
    #     self.size = len(src_data) if src_data is not None else len(tgt_data)
    #
    #     if src_lang_data is not None:
    #         self.tensors['source_lang'] = torch.cat(src_lang_data).long()   # langage ID per sentence
    #
    #         if token_level_lang:  # prepare token-level language prediction targets
    #             sl_ = self.tensors['source_lang']  # sl_.shape[0]
    #             out_dims = (max(self.src_lengths).item(), sl_.shape[0])  # T, B
    #             out_tensor = sl_.data.new(*out_dims).fill_(onmt.constants.PAD)
    #             for i, v in enumerate(sl_):
    #                 # lan ID starts with 0, but pred label values should start with 1 (due to padding)
    #                 out_tensor[:(self.src_lengths[i]), i] = v+1
    #
    #             self.tensors['targets_source_lang'] = out_tensor
    #
    #     if tgt_lang_data is not None:
    #         self.tensors['target_lang'] = torch.cat(tgt_lang_data).long()

    def switchout(self, swrate, src_vocab_size, tgt_vocab_size):
        # Switch out function ... currently works with only source text data
        # if self.src_type == 'text':
        if len(self.tensors['source'].shape) == 2:
            self.tensors['source'] = switchout(self.tensors['source'], src_vocab_size, swrate, transpose=True)

        if self.has_target:
            self.tensors['target'] = switchout(self.tensors['target'], tgt_vocab_size, swrate, transpose=True, offset=1)
            target_full = self.tensors['target']
            self.tensors['target_input'] = target_full[:-1]
            self.tensors['target_output'] = target_full[1:]
            self.tensors['tgt_mask'] = self.tensors['target_output'].ne(onmt.constants.PAD)

    def collate(self, data, align_right=False, type="text", augmenter=None):
        """
        Assembling the individual sequences into one single tensor, included padding
        :param data: the list of sequences
        :param align_right: aligning the sequences w.r.t padding
        :param type: text or audio
        :param augmenter: for augmentation in audio models
        :return: tensor: tensor of source token IDs, padded
        :return: pos: tensor of position counters, padded
        :return: lengths: array of source length
        """
        # initialize with batch_size * length
        if type == "text":
            lengths = [x.size(0) for x in data]
            positions = [torch.arange(length_) for length_ in lengths]
            max_length = max(lengths)
            tensor = data[0].new(len(data), max_length).fill_(onmt.constants.PAD)
            pos = tensor.new(*tensor.size()).fill_(0)

            for i in range(len(data)):
                data_length = data[i].size(0)
                offset = max_length - data_length if align_right else 0
                tensor[i].narrow(0, offset, data_length).copy_(data[i])
                pos[i].narrow(0, offset, data_length).copy_(positions[i])

            return tensor, pos, lengths

        elif type == "audio":

            # First step: on-the-fly processing for the samples
            # Reshaping: either downsampling or upsampling
            # On the fly augmentation
            samples = []

            for i in range(len(data)):
                sample = data[i]

                if augmenter is not None:
                    sample = augmenter.augment(sample)

                if self.upsampling:
                    sample = sample.view(-1, self.feature_size)

                samples.append(sample)

            # compute the lengths afte on-the-fly processing
            lengths = [x.size(0) for x in samples]

            max_length = max(lengths)

            # allocate data for the batch speech
            feature_size = samples[0].size(1)
            batch_size = len(data)

            # feature size + 1 because the last dimension is created for padding
            tensor = data[0].float().new(batch_size, max_length, feature_size + 1).fill_(onmt.constants.PAD)

            for i in range(len(samples)):
                sample = samples[i]

                data_length = sample.size(0)
                offset = max_length - data_length if align_right else 0

                tensor[i].narrow(0, offset, data_length).narrow(1, 1, sample.size(1)).copy_(sample)
                # in padding dimension: 0 is not padded, 1 is padded
                tensor[i].narrow(0, offset, data_length).narrow(1, 0, 1).fill_(1)

            return tensor, None, lengths
        else:
            raise NotImplementedError


class LightBatch:

    def __init__(self, tensors):
        self.tensors = tensors

    def pin_memory(self):
        """
        Enable memory pinning
        :return:
        """
        for key, tensor in self.tensors.items():
            if isinstance(tensor, dict):
                for k in tensor:
                    v = tensor[k]
                    if isinstance(v, torch.Tensor):
                        tensor[k] = v.pin_memory()
            elif tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    self.tensors[key] = self.tensors[key].pin_memory()
            else:
                continue
        return self


class Dataset(torch.utils.data.Dataset):
    def __init__(self, src_data, tgt_data,
                 src_sizes=None, tgt_sizes=None,
                 src_langs=None, tgt_langs=None,
                 batch_size_words=16384,
                 data_type="text", batch_size_sents=128,
                 multiplier=1, sorting=False,
                 augment=False,
                 src_align_right=False, tgt_align_right=False,
                 verbose=False, cleaning=False, debug=False,
                 num_split=1,
                 token_level_lang=False,
                 bidirectional=False,
                 multidataset=False,
                 en_id=None,
                 **kwargs):
        """
        :param src_data: List of tensors for the source side (1D for text, 2 or 3Ds for other modalities)
        :param tgt_data: List of tensors (1D text) for the target side (already padded with <s> and </s>
        :param src_langs: Source languages (list of one-tensors)
        :param tgt_langs: Target Languages (list of one-tensors)
        :param batch_size_words: Maximum number of words in the minibatch (MB can't have more than this)
        :param data_type: Text or Audio
        :param batch_size_sents: Maximum number of sequences in the minibatch (MB can't have more than this)
        :param multiplier: The number of sequences must divide by this number (for fp16 when multiplier=8)
        :param reshape_speech: Put N frames together to reduce the length (this might be done already in preprocessing)
        :param augment: Speech Augmentation (currently only spec augmentation is implemented)
        """

        """
        For alignment, the right-aligned data looks like:
        P P P P D D D D
        P P D D D D D D
        P P P P P D D D
        P P P D D D D D
        This can affect positional encoding (whose implementation is not consistent w.r.t padding)
        For models with absolute positional encoding, src and tgt should be aligned left (This is default)
        For models with relative positional encoding, src should be right and tgt should be left
        """

        self.src = src_data
        self._type = data_type
        self.src_align_right = src_align_right
        if self.src_align_right and verbose:
            print("* Source sentences aligned to the right side.")
        self.tgt_align_right = tgt_align_right
        self.upsampling = kwargs.get('upsampling', False)

        self.max_src_len = kwargs.get('max_src_len', None)
        self.max_tgt_len = kwargs.get('max_tgt_len', 256)
        self.cleaning = cleaning
        self.debug = debug
        self.num_split = num_split
        self.vocab_mask = None

        if self.max_src_len is None:
            if self._type == 'text':
                self.max_src_len = 256
            else:
                self.max_src_len = 1024

        # self.reshape_speech = reshape_speech
        if tgt_data:
            self.tgt = tgt_data

            if src_data:
                assert (len(self.src) == len(self.tgt))
        else:
            self.tgt = None

        self.order = np.arange(len(self.src))

        # Processing data sizes
        if self.src is not None:
            if src_sizes is not None:
                self.src_sizes = np.asarray(src_sizes)
            else:
                self.src_sizes = np.asarray([data.size(0) for data in self.src])
        else:
            self.src_sizes = None

        if self.tgt is not None:
            if tgt_sizes is not None:
                self.tgt_sizes = np.asarray(tgt_sizes)
            else:
                self.tgt_sizes = np.asarray([data.size(0) for data in self.tgt])
        else:
            self.tgt_sizes = None

        # sort data to have efficient mini-batching during training
        if sorting:
            if verbose:
                print("* Sorting data ...")

            if self._type == 'text':
                sorted_order = np.lexsort((self.src_sizes, self.tgt_sizes))
            elif self._type == 'audio':
                sorted_order = np.lexsort((self.tgt_sizes, self.src_sizes))

            self.order = sorted_order

        # store data length in numpy for fast query
        if self.tgt is not None and self.src is not None:
            stacked_sizes = np.stack((self.src_sizes, self.tgt_sizes - 1), axis=0)
            self.data_lengths = np.amax(stacked_sizes, axis=0)
        elif self.src is None:
            self.data_lengths = self.tgt_sizes
        else:
            self.data_lengths = self.src_sizes
        # sort data to have efficient mini-batching during training
        # if sorting:
        #     if verbose:
        #         print("* Sorting data ...")
        #     if self._type == 'text':
        #         sorted_order = np.lexsort((self.src_sizes, self.tgt_sizes))
        #     elif self._type == 'audio':
        #         sorted_order = np.lexsort((self.tgt_sizes, self.src_sizes))
        #
        #     self.order = sorted_order

            # # store data length in numpy for fast query
            # if self.tgt is not None and self.src is not None:
            #     stacked_sizes = np.stack((self.src_sizes, self.tgt_sizes - 1), axis=0)
            #     self.data_lengths = np.amax(stacked_sizes, axis=0)
            # elif self.src is None:
            #     self.data_lengths = self.tgt_sizes
            # else:
            #     self.data_lengths = self.src_sizes

        # Processing language ids
        self.src_langs = src_langs
        self.tgt_langs = tgt_langs

        if self.src_langs is not None and self.tgt_langs is not None:
            assert (len(src_langs) == len(tgt_langs))

        # In "bilingual" case, the src_langs only contains one single vector
        # Which is broadcasted to batch_size
        if len(src_langs) <= 1:
            self.bilingual = True
        else:
            self.bilingual = False

        self.full_size = len(self.src) if self.src is not None else len(self.tgt)

        # maximum number of tokens in a mb
        self.batch_size_words = batch_size_words

        # maximum sequences in a mb
        self.batch_size_sents = batch_size_sents

        # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        self.multiplier = multiplier

        # by default: count the amount of padding when we group mini-batches
        self.pad_count = True

        # group samples into mini-batches
        if verbose:
            print("* Allocating mini-batches ...")
        self.batches = allocate_batch(self.order, self.data_lengths,
                                      self.src_sizes, self.tgt_sizes,
                                      batch_size_words, batch_size_sents, self.multiplier,
                                      self.max_src_len, self.max_tgt_len, self.cleaning)

        # the second to last mini-batch is likely the largest
        # (the last one can be the remnant after grouping samples which has less than max size)
        self.largest_batch_id = len(self.batches) - 2

        self.num_batches = len(self.batches)

        self.cur_index = 0
        self.batchOrder = None

        if augment:
            self.augmenter = Augmenter()
        else:
            self.augmenter = None
        #
        #     assert self.src is not None
        #     assert self.tgt is not None
        #
        #     src_sizes = [data.size(0) for data in self.src]
        #     tgt_sizes = [data.size(0) for data in self.tgt]
        #     orders = range(len(self.src))
        #
        #     z = zip(src_sizes, tgt_sizes, orders)
        #
        #     if self._type == 'text':
        #         # For machine translation, sort by source first, and then target
        #         sorted_z = sorted(sorted(z, key=lambda x: x[0]), key=lambda x: x[1])
        #
        #     elif self._type == 'audio':
        #         sorted_z = sorted(sorted(z, key=lambda x: x[1]), key=lambda x: x[0])
        #
        #     sorted_order = [z_[2] for z_ in sorted_z]
        #
        #     self.src = [self.src[i] for i in sorted_order]
        #     self.tgt = [self.tgt[i] for i in sorted_order]
        #
        # self.src_langs = src_langs
        # self.tgt_langs = tgt_langs
        #
        # if self.src_langs is not None and self.tgt_langs is not None:
        #     assert(len(src_langs) == len(tgt_langs))
        #
        # # In "bilingual" case, the src_langs only contains one single vector
        # # Which is broadcasted to batch_size
        # if len(src_langs) <= 1:
        #     self.bilingual = True
        # else:
        #     self.bilingual = False
        #     if sorting:
        #         self.src_langs = [self.src_langs[i] for i in sorted_order]
        #         self.tgt_langs = [self.tgt_langs[i] for i in sorted_order]
        #
        # self.fullSize = len(self.src) if self.src is not None else len(self.tgt)
        #
        # # maximum number of tokens in a mb
        # self.batch_size_words = batch_size_words
        #
        # # maximum sequences in a mb
        # self.batch_size_sents = batch_size_sents
        #
        # # the actual batch size must divide by this multiplier (for fp16 it has to be 4 or 8)
        # self.multiplier = multiplier
        #
        # # by default: count the amount of padding when we group mini-batches
        # self.pad_count = True
        #
        # # group samples into mini-batches
        # self.batches = []
        # self.num_batches = 0
        # self.allocate_batch()
        #
        # self.cur_index = 0
        # self.batchOrder = None
        #
        # if augment:
        #     self.augmenter = Augmenter()
        # else:
        #     self.augmenter = None

        self.token_level_lang = token_level_lang
        self.bidirectional_translation = bidirectional
        self.multidataset = multidataset
        self.en_id = en_id

    def size(self):

        return self.full_size

    def switchout(self, batch):

        pass

    def set_epoch(self, epoch):

        pass

    def set_mask(self, vocab_mask):
        self.vocab_mask = vocab_mask

    def get_largest_batch(self):

        return self.get_batch(self.largest_batch_id)

    # This function allocates the mini-batches (grouping sentences with the same size)
    # def allocate_batch(self):
    #
    #     cur_batch = []
    #     cur_batch_size = 0
    #     cur_batch_sizes = []
    #
    #     def oversize_(cur_batch, sent_size):
    #
    #         if len(cur_batch) >= self.batch_size_sents:
    #             return True
    #
    #         if not self.pad_count:
    #             if cur_batch_size + sent_size > self.batch_size_words:
    #                 return True
    #         else:
    #             if len(cur_batch_sizes) == 0:
    #                 return False
    #
    #             if (max(max(cur_batch_sizes), sent_size)) * (len(cur_batch) + 1) > self.batch_size_words:
    #                 return True
    #         return False
    #
    #     i = 0
    #     while i < self.fullSize:
    #
    #         if self.tgt is not None and self.src is not None:
    #             sentence_length = max(self.tgt[i].size(0) - 1, self.src[i].size(0))
    #             # print(sentence_length)
    #         elif self.tgt is not None:
    #             sentence_length = self.tgt[i].size(0) - 1
    #         else:
    #             sentence_length = self.src[i].size(0)
    #
    #         oversized = oversize_(cur_batch, sentence_length)
    #         # if the current item makes the batch exceed max size
    #         # then we create a new batch
    #         if oversized:
    #             # cut-off the current list to fit the multiplier
    #             current_size = len(cur_batch)
    #             scaled_size = max(
    #                 self.multiplier * (current_size // self.multiplier),
    #                 current_size % self.multiplier)
    #
    #             batch_ = cur_batch[:scaled_size]
    #             self.batches.append(batch_)  # add this batch into the batch list
    #
    #             cur_batch = cur_batch[scaled_size:]  # reset the current batch
    #             cur_batch_sizes = cur_batch_sizes[scaled_size:]
    #             cur_batch_size = sum(cur_batch_sizes)
    #
    #         cur_batch.append(i)
    #         cur_batch_size += sentence_length
    #         cur_batch_sizes.append(sentence_length)
    #
    #         i = i + 1
    #
    #     # catch the last batch
    #     if len(cur_batch) > 0:
    #         self.batches.append(cur_batch)
    #
    #     self.num_batches = len(self.batches)

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        src_lang, tgt_lang = None, None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang = self.src_langs[0]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[0]  # should be a tensor [1]
        else:
            if self.src_langs is not None:
                src_lang = self.src_langs[index]
            if self.tgt_langs is not None:
                tgt_lang = self.tgt_langs[index]

        # move augmenter here?

        sample = {
            'src': self.src[index] if self.src is not None else None,
            'tgt': self.tgt[index] if self.tgt is not None else None,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang
        }

        return sample

        # assert index < self.num_batches, "%d > %d" % (index, self.num_batches)
        #
        # batch_ids = self.batches[index]
        # if self.src:
        #     src_data = [self.src[i] for i in batch_ids]
        # else:
        #     src_data = None
        #
        # if self.tgt:
        #     tgt_data = [self.tgt[i] for i in batch_ids]
        # else:
        #     tgt_data = None
        #
        # src_lang_data = None
        # tgt_lang_data = None
        #
        # if self.bilingual:
        #     if self.src_langs is not None:
        #         src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
        #     if self.tgt_langs is not None:
        #         tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
        # else:
        #     if self.src_langs is not None:
        #         src_lang_data = [self.src_langs[i] for i in batch_ids]
        #     if self.tgt_langs is not None:
        #         tgt_lang_data = [self.tgt_langs[i] for i in batch_ids]
        #
        # batch = Batch(src_data, tgt_data=tgt_data,
        #               src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
        #               src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
        #               src_type=self._type,
        #               augmenter=self.augmenter, upsampling=self.upsampling,
        #               token_level_lang=self.token_level_lang)
        #
        # return batch

    def get_batch(self, index):
        """
        This function is only used in when we need to access a batch directly from the dataset
        (Without an external loader)
        :param index: the index of the mini-batch in the list
        :return: Batch
        """
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)

        batch_ids = self.batches[index]
        if self.src:
            src_data = [self.src[i] for i in batch_ids]
        else:
            src_data = None

        if self.tgt:
            tgt_data = [self.tgt[i] for i in batch_ids]
        else:
            tgt_data = None

        src_lang_data = None
        tgt_lang_data = None

        if self.bilingual:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
        else:
            if self.src_langs is not None:
                src_lang_data = [self.src_langs[i] for i in batch_ids]
            if self.tgt_langs is not None:
                tgt_lang_data = [self.tgt_langs[i] for i in batch_ids]

        batch = rewrap(collect_fn(src_data, tgt_data=tgt_data,
                                  src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                                  src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                                  src_type=self._type,
                                  augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                                  token_level_lang=self.token_level_lang,
                                  bidirectional=self.bidirectional_translation,
                                  multidataset=self.multidataset,
                                  en_id=self.en_id)
                       )
        return batch

    def collater(self, collected_samples):
        """
        Merge a list of samples into a Batch
        :param collected_samples: list of dicts (the output of the __getitem__)
        :return: batch
        """

        split_size = math.ceil(len(collected_samples) / self.num_split)
        sample_list = [collected_samples[i:i+split_size]
                       for i in range(0, len(collected_samples), split_size)]

        batches = list()

        for samples in sample_list:

            src_data, tgt_data = None, None
            src_lang_data, tgt_lang_data = None, None

            if self.src:
                src_data = [sample['src'] for sample in samples]

            if self.tgt:
                tgt_data = [sample['tgt'] for sample in samples]

            if self.bilingual:
                if self.src_langs is not None:
                    src_lang_data = [self.src_langs[0]]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [self.tgt_langs[0]]  # should be a tensor [1]
            else:
                if self.src_langs is not None:
                    src_lang_data = [sample['src_lang'] for sample in samples]  # should be a tensor [0]
                if self.tgt_langs is not None:
                    tgt_lang_data = [sample['tgt_lang'] for sample in samples]  # should be a tensor [1]

            batch = collect_fn(src_data, tgt_data=tgt_data,
                               src_lang_data=src_lang_data, tgt_lang_data=tgt_lang_data,
                               src_align_right=self.src_align_right, tgt_align_right=self.tgt_align_right,
                               src_type=self._type,
                               augmenter=self.augmenter, upsampling=self.upsampling, vocab_mask=self.vocab_mask,
                               token_level_lang=self.token_level_lang,
                               bidirectional=self.bidirectional_translation,
                               multidataset=self.multidataset,
                               en_id=self.en_id)

            batches.append(batch)

        return batches

    def __len__(self):
        return self.full_size

    # genereate a new batch - order (static)
    def create_order(self, random=True):

        if random:
            self.batchOrder = torch.randperm(self.num_batches)
        else:
            self.batchOrder = torch.arange(self.num_batches).long()

        self.cur_index = 0

        return self.batchOrder

    # return the next batch according to the iterator
    def next(self, curriculum=False, reset=True):

        # reset iterator if reach data size limit
        if self.cur_index >= self.num_batches:
            if reset:
                self.cur_index = 0
            else:
                return None

        if curriculum or self.batchOrder is None:
            batch_index = self.cur_index
        else:
            batch_index = self.batchOrder[self.cur_index]

        batch = self[batch_index]

        # move the iterator one step
        self.cur_index += 1

        return [batch]

    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        self.src, self.tgt = zip(*[data[i] for i in torch.randperm(len(data))])

    def set_index(self, iteration):

        assert (0 <= iteration < self.num_batches)
        self.cur_index = iteration

#
# # LANGUAGE MODEL DATASET AND DATAHOLDER
# class LMBatch(Batch):
#
#     def __init__(self, input, target=None):
#         self.tensors = defaultdict(lambda: None)
#
#         self.tensors['target_input'] = input  # T x B
#         self.tensors['target_output'] = target  # T x B or None
#
#         # batch size
#         self.size = input.size(1)
#         self.length = input.size(0)
#
#         self.tgt_size = self.size * self.length
#         self.src_size = 0
#
#     def collate(self, **kwargs):
#         raise NotImplementedError
#
#
# class LanguageModelDataset(Dataset):
#
#     def __init__(self, data, batch_size_sents=128, seq_length=128):
#
#         self.data = data
#
#         self.batch_size_sents = batch_size_sents
#
#         self.seq_length = seq_length
#
#         # group samples into mini batches
#         self.num_batches = 0
#         self.allocate_batch()
#
#         self.fullSize = self.num_batches
#         # self.cur_index = 0
#         # self.batchOrder = None
#
#     def allocate_batch(self):
#
#         nsequence = self.data.size(0) // self.batch_size_sents
#
#         self.data = self.data.narrow(0, 0, nsequence * self.batch_size_sents)
#
#         # Evenly divide the data across the bsz batches.
#         self.data = self.data.view(self.batch_size_sents, -1).t().contiguous()
#
#         # self.num_steps = nbatch - 1
#
#         self.num_batches = math.ceil((self.data.size(0) - 1) / self.seq_length)
#
#     # genereate a new batch - order (static)
#     def create_order(self, random=False):
#
#         # For language model order shouldn't be random
#         if random:
#             self.batchOrder = torch.randperm(self.num_batches)
#         else:
#             self.batchOrder = torch.arange(self.num_batches).long()
#
#         self.cur_index = 0
#
#         return self.batchOrder
#
#     # return the next batch according to the iterator
#     # for language model
#     def next(self, curriculum=True, reset=True, split_sizes=1):
#
#         # reset iterator if reach data size limit
#         if self.cur_index >= self.num_batches:
#             if reset:
#                 self.cur_index = 0
#             else:
#                 return None
#
#         batch_index = self.cur_index
#
#         seq_len = self.seq_length
#
#         top_index = min(batch_index + seq_len, self.data.size(0) - 1)
#
#         batch = LMBatch(self.data[batch_index:top_index], target=self.data[batch_index + 1:top_index + 1])
#
#         # move the iterator one step
#         self.cur_index += seq_len
#
#         return [batch]