import onmt
import onmt.modules
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from onmt.model_factory import build_model
import torch.nn.functional as F
from onmt.inference.search import BeamSearch, DiverseBeamSearch
from onmt.inference.translator import Translator
import sys
model_list = ['transformer', 'stochastic_transformer']


class CTCTranslator(Translator):
    def __init__(self, opt):
        super().__init__(opt)
        self.search = BeamSearch(self.tgt_dict)
        self.eos = onmt.constants.EOS
        self.pad = onmt.constants.PAD
        self.bos = self.bos_id
        self.vocab_size = self.tgt_dict.size()
        self.min_len = 1
        self.normalize_scores = opt.normalize
        self.len_penalty = opt.alpha
        self.buffering = not opt.no_buffering

        if hasattr(opt, 'no_repeat_ngram_size'):
            self.no_repeat_ngram_size = opt.no_repeat_ngram_size
        else:
            self.no_repeat_ngram_size = 0

        if hasattr(opt, 'dynamic_max_len'):
            self.dynamic_max_len = opt.dynamic_max_len
        else:
            self.dynamic_max_len = False

        if hasattr(opt, 'dynamic_max_len_scale'):
            self.dynamic_max_len_scale = opt.dynamic_max_len_scale
        else:
            self.dynamic_max_len_scale = 1.2

        if opt.verbose:
            print('* Current bos id: %d' % self.bos_id, onmt.constants.BOS)
            print('* Using fast beam search implementation')

    def translateBatch(self, batch):

        with torch.no_grad():
            return self._translateBatch(batch)

    def _translateBatch(self, batch):

        # Batch size is in different location depending on data.

        beam_size = self.opt.beam_size
        bsz = batch_size = batch.size

        max_len = self.opt.max_sent_length

        gold_scores = batch.get('source').data.new(batch_size).float().zero_()
        gold_words = 0
        allgold_scores = []

        if batch.has_target:
            # Use the first model to decode
            model_ = self.models[0]

            gold_words, gold_scores, allgold_scores = model_.decode(batch)

        # Use the first model to decode
        model = self.models[0]
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        src_lang = batch.get('source_lang')

        src_transposed = src.transpose(0, 1)
        # enc output
        encoder_output = model.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang)
        # encoder logits: T x B x V
        logits = model.generator[0](encoder_output, input_name='context')
        # T, _, H = logits.shape

        # T x B x V --> T x B
        max_prob_indices = torch.argmax(logits, dim=2)

        finalized = []
        for i in range(batch_size):
            # T --> T' (shrunken by removing blank and repetition)
            sample_logit = max_prob_indices[:, i]
            hyp = torch.unique_consecutive(sample_logit[sample_logit != onmt.constants.PAD])
            finalized.append(hyp)

        return finalized, gold_scores, gold_words, allgold_scores

    def _decode(self, tokens, decoder_states):

        # require batch first for everything
        outs = dict()
        attns = dict()

        for i in range(self.n_models):
            decoder_output = self.models[i].step(tokens, decoder_states[i])

            # take the last decoder state
            # decoder_hidden = decoder_hidden.squeeze(1)
            # attns[i] = coverage[:, -1, :].squeeze(1)  # batch * beam x src_len

            # batch * beam x vocab_size
            # outs[i] = self.models[i].generator(decoder_hidden)
            outs[i] = decoder_output['log_prob']
            attns[i] = decoder_output['coverage']

        out = self._combine_outputs(outs)
        # attn = self._combine_attention(attns)

        if self.vocab_size > out.size(-1):
            self.vocab_size = out.size(-13)
        # attn = attn[:, -1, :] # I dont know what this line means
        attn = None # lol this is never used probably

        return out, attn

    def translate(self, src_data, tgt_data, type='mt'):
        #  (1) convert words to indices
        dataset = self.build_data(src_data, tgt_data, type=type)
        batch = dataset.get_batch(0)
        # batch = dataset.next()[0]
        if self.cuda:
            batch.cuda(fp16=self.fp16)
        # ~ batch = self.to_variable(dataset.next()[0])
        batch_size = batch.size

        #  (2) translate
        finalized, gold_score, gold_words, allgold_words = self.translateBatch(batch)
        pred_length = []

        #  (3) convert indices to words
        pred_batch = []
        for b in range(batch_size):
            pred_batch.append(
                [self.tgt_dict.convertToLabels(finalized[b], stop=onmt.constants.EOS, including_stop=False)]
            )
        # self.build_target_tokens(finalized[b][n]['tokens'], src_data[b], None)

        # TODO: pred score is different!
        pred_score = []
        for b in range(batch_size):
            pred_score.append(
                [torch.FloatTensor([0.0]) # [finalized[b][n]['score']]
                 for n in range(self.opt.n_best)]
            )

        return pred_batch, pred_score, pred_length, gold_score, gold_words, allgold_words

