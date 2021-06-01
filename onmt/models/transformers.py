import copy
import math
import numpy as np
import torch, math
import torch.nn as nn
from onmt.models.transformer_layers import EncoderLayer, DecoderLayer, PositionalEncoding, \
    PrePostProcessing
from onmt.modules.base_seq2seq import NMTModel, Reconstructor, DecoderState
import onmt
from onmt.modules.dropout import embedded_dropout, switchout
from torch.utils.checkpoint import checkpoint
from collections import defaultdict
from onmt.utils import flip, expected_length
from onmt.modules.linear import FeedForward, FeedForwardSwish
import copy

torch_version = float(torch.__version__[:3])


class MixedEncoder(nn.Module):

    def __init(self, text_encoder, audio_encoder):
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

    def forward(self, input, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (to be transposed)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """

        if input.dim() == 2:
            return self.text_encoder.forward(input)
        else:
            return self.audio_encoder.forward(input)


class MultiSourceEncoder(nn.Module):
    def __init__(self, opt, embedding, encoder_main, encoder_aux, encoder_type='text', language_embeddings=None):
        super(MultiSourceEncoder, self).__init__()

        self.encoder_main = encoder_main
        self.encoder_aux = encoder_aux

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout

        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling
        self.death_rate = opt.death_rate

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time

        # disable word dropout when switch out is in action
        if self.switchout > 0.0:
            self.word_dropout = 0.0

        feature_size = opt.input_size
        self.channels = 1  # n. audio channels

        if opt.upsampling:
            feature_size = feature_size // 4

        # if encoder_type != "text":
        #     if not self.cnn_downsampling:
        #         self.audio_trans = nn.Linear(feature_size, self.model_size)
        #         torch.nn.init.xavier_uniform_(self.audio_trans.weight)
        #     else:
        #         channels = self.channels
        #         cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
        #                nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]
        #
        #         feat_size = (((feature_size // channels) - 3) // 4) * 32
        #         # cnn.append()
        #         self.audio_trans = nn.Sequential(*cnn)
        #         self.linear_trans = nn.Linear(feat_size, self.model_size)
                # assert self.model_size == feat_size, \
                #     "The model dimension doesn't match with the feature dim, expecting %d " % feat_size
        # else:
        self.word_lut = embedding

        # if opt.time == 'positional_encoding':
        #     self.time_transformer = positional_encoder
        # elif opt.time == 'gru':
        #     self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        # elif opt.time == 'lstm':
        #     self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        # self.time_transformer = positional_encoder
        # self.language_embedding = language_embeddings
        #
        # self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
        #                                           variational=self.varitional_dropout)
        #
        # self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        # self.positional_encoder = positional_encoder
        self.do_mixing = opt.mix_encoder_outputs
        self.enc_out_type = opt.enc_out_type
        self.language_classifier = opt.language_classifier

    def forward(self, input, input_lang=None, **kwargs):
        if isinstance(input, list):  #self.training:  # training time, randomly choose
            # print('*** Random selection.')
            lan_main, lan_aux = None, None
#            if len(input_lang.shape)==3:
            lan_main = input_lang[0]
            lan_aux = input_lang[1]
            enc_out_main = self.encoder_main.forward(input[0], input_lang=lan_main)  # take main-language input
            enc_out_aux = self.encoder_aux.forward(input[1], input_lang=lan_aux)  # take aux-language input
            # context is T x B x H, e.g. 10, 128, 512
            context_main = enc_out_main['context']
            context_aux = enc_out_aux['context']
            src_mask_main = enc_out_main['src_mask']
            src_mask_aux = enc_out_aux['src_mask']

            output_dict = {'context': context_main, 'src_mask': src_mask_main, 'context_main': context_main, 'context_aux': context_aux, 'src_mask_main': src_mask_main, 'src_mask_aux': src_mask_aux}
            return output_dict
        else:  #when we don't have multiway src. # valid/test time, only take main encoder (We don't necessarily have multiway dev data!)
            if input_lang is not None:
                return self.encoder_main.forward(input, input_lang=input_lang)
            else:
                return self.encoder_main.forward(input)


class TransformerEncoder(nn.Module):
    """Encoder in 'Attention is all you need'

    Args:
        opt: list of options ( see train.py )
        dicts : dictionary (for source language)

    """

    def __init__(self, opt, embedding, positional_encoder, encoder_type='text', language_embeddings=None):

        super(TransformerEncoder, self).__init__()

        self.opt = opt
        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        if hasattr(opt, 'encoder_layers') and opt.encoder_layers != -1:
            self.layers = opt.encoder_layers
        else:
            self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout

        self.input_type = encoder_type
        self.cnn_downsampling = opt.cnn_downsampling
        self.death_rate = opt.death_rate

        self.switchout = opt.switchout
        self.varitional_dropout = opt.variational_dropout
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        self.time = opt.time

        # disable word dropout when switch out is in action
        if self.switchout > 0.0:
            self.word_dropout = 0.0

        feature_size = opt.input_size
        self.channels = 1  # n. audio channels

        if opt.upsampling:
            feature_size = feature_size // 4

        if encoder_type != "text":
            if not self.cnn_downsampling:
                self.audio_trans = nn.Linear(feature_size, self.model_size)
                torch.nn.init.xavier_uniform_(self.audio_trans.weight)
            else:
                channels = self.channels
                cnn = [nn.Conv2d(channels, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32),
                       nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2), nn.ReLU(True), nn.BatchNorm2d(32)]

                feat_size = (((feature_size // channels) - 3) // 4) * 32
                # cnn.append()
                self.audio_trans = nn.Sequential(*cnn)
                self.linear_trans = nn.Linear(feat_size, self.model_size)
                # assert self.model_size == feat_size, \
                #     "The model dimension doesn't match with the feature dim, expecting %d " % feat_size
        else:
            self.word_lut = embedding

        self.change_residual_at = opt.change_residual_at
        self.change_residual = None if self.change_residual_at is None else opt.change_residual
        self.change_att_query_at = opt.change_att_query_at
        self.change_att_query = None if self.change_att_query_at is None else opt.change_att_query
        self.att_plot_path = None
        self.save_activation = None
        # if opt.time == 'positional_encoding':
        #     self.time_transformer = positional_encoder
        # elif opt.time == 'gru':
        #     self.time_transformer = nn.GRU(self.model_size, self.model_size, 1, batch_first=True)
        # elif opt.time == 'lstm':
        #     self.time_transformer = nn.LSTM(self.model_size, self.model_size, 1, batch_first=True)
        self.time_transformer = positional_encoder
        self.language_embedding = language_embeddings

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.varitional_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.positional_encoder = positional_encoder

        self.layer_modules = nn.ModuleList()
        self.build_modules(opt)

        if opt.freeze_encoder:
            self.requires_grad_(False)

        self.language_classifier = opt.language_classifier
        self.language_classifier_sent = opt.language_classifier_sent

        self.token_classifier_at = opt.token_classifier_at
        self.language_specific_encoder = opt.language_specific_encoder


    def build_modules(self, opt):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Encoder with Absolute Attention with %.2f expected layers" % e_length)

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            change_residual = None
            change_residual_here = (self.change_residual_at == 0) or \
                                   (self.change_residual_at == _l + 1) or \
                                   (self.change_residual_at == -1 and _l == self.layers - 1)
            if change_residual_here:
                print('*** Layer (indexing from 0)', _l, 'change residual to code', self.change_residual)
                change_residual = self.change_residual
            change_att_query = None
            change_att_query_here = (self.change_att_query_at == 0) or \
                                    (self.change_att_query_at == _l + 1) or \
                                    (self.change_att_query_at == -1 and _l == self.layers - 1)
            if change_att_query_here:
                print('*** Layer (indexing from 0)', _l, 'change att query to code', self.change_att_query)
                change_att_query = self.change_att_query

            if opt.language_specific_encoder:
                if opt.language_specific_encoder == 1:  # 1 used to mean all
                    add_adapter = True
                elif _l + 1 in opt.language_specific_encoder:
                    add_adapter = True
                else:
                    add_adapter = False
            else:
                add_adapter = False

            if add_adapter:
                print('*** Layer (indexing from 0)', _l, '+ adapter')

            block = EncoderLayer(self.n_heads, self.model_size,
                                 self.dropout, self.inner_size, self.attn_dropout,
                                 variational=self.varitional_dropout, death_rate=death_r,
                                 change_residual=change_residual,
                                 change_att_query=change_att_query,
                                 add_adapter=add_adapter,
                                 residual_death_rate=(_l + 1.0) / self.layers * 0.5,
                                 opt=opt)

            self.layer_modules.append(block)

        # if language_specific_encoder == number of encoder layers + 1:
        # put adapter at the very last (after the final layer norm)
        if opt.language_specific_encoder and (opt.language_specific_encoder[0] == self.layers+1):
            print('*** After full encoder + adapter')
            self.add_adapter = True
            adapter_bottleneck_size = opt.model_size // 2
            from onmt.modules.multilingual_factorized.multilingual_adapters import MultilingualAdapter
            self.adapters = MultilingualAdapter(model_size=opt.model_size, bottleneck_size=adapter_bottleneck_size,
                                                n_languages=opt.n_languages,
                                                dropout=opt.dropout, variational=opt.variational_dropout)
        else:
            self.add_adapter = False

        # self.layer_modules = nn.ModuleList(
        #     [EncoderLayer(self.n_heads, self.model_size, self.dropout, self.inner_size,
        #                   self.attn_dropout, variational=self.varitional_dropout) for _ in
        #      range(self.layers)])

    def forward(self, input, input_lang=None, **kwargs):
        """
        Inputs Shapes:
            input: batch_size x len_src (to be transposed)

        Outputs Shapes:
            out: batch_size x len_src x d_model
            mask_src

        """

        """ Embedding: batch_size x len_src x d_model """
        if self.input_type == "text":
            mask_src = input.eq(onmt.constants.PAD).unsqueeze(1)  # batch_size x 1 x len_src for broadcasting

            # apply switchout
            # if self.switchout > 0 and self.training:
            #     vocab_size = self.word_lut.weight.size(0)
            #     input = switchout(input, vocab_size, self.switchout)

            emb = embedded_dropout(self.word_lut, input, dropout=self.word_dropout if self.training else 0)
        else:
            if not self.cnn_downsampling:
                mask_src = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                input = input.narrow(2, 1, input.size(2) - 1)
                emb = self.audio_trans(input.contiguous().view(-1, input.size(2))).view(input.size(0),
                                                                                        input.size(1), -1)
                emb = emb.type_as(input)
            else:
                long_mask = input.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                input = input.narrow(2, 1, input.size(2) - 1)

                # first resizing to fit the CNN format
                input = input.view(input.size(0), input.size(1), -1, self.channels)
                input = input.permute(0, 3, 1, 2)

                input = self.audio_trans(input)
                input = input.permute(0, 2, 1, 3).contiguous()
                input = input.view(input.size(0), input.size(1), -1)
                # print(input.size())
                input = self.linear_trans(input)

                mask_src = long_mask[:, 0:input.size(1) * 4:4].unsqueeze(1)
                # the size seems to be B x T ?
                emb = input

        if torch_version >= 1.2:
            mask_src = mask_src.bool()

        """ Scale the emb by sqrt(d_model) """
        emb = emb * math.sqrt(self.model_size)

        """ Adding positional encoding """
        emb = self.time_transformer(emb)
        if self.change_att_query is None:
            given_query = None
        else:
            given_query = self.time_transformer(emb, change_code=self.change_att_query).transpose(0, 1)

        """ Adding language embeddings """
        if self.use_language_embedding:
            assert self.language_embedding is not None

            if self.language_embedding_type in ['sum', 'all_sum']:
                lang_emb = self.language_embedding(input_lang)
                emb = emb + lang_emb.unsqueeze(1)

        # B x T x H -> T x B x H
        context = emb.transpose(0, 1)

        context = self.preprocess_layer(context)

        output_dict = {}

        for i, layer in enumerate(self.layer_modules):
            context = layer(context, mask_src, given_query=given_query, att_plot_path=self.att_plot_path,
                            src_lang=input_lang)   # batch_size x len_src x d_model

            # If save_activation is not none and we're at last encoder layer
            if self.save_activation is not None and (i == len(self.layer_modules) - 1):
                # Zero-pad layer output
                padded_context = context.masked_fill(mask_src.permute(2, 0, 1), 0).type_as(context)
                try:
                    # Load existing file
                    saved_att = torch.load(self.save_activation)
                except OSError:
                    # In case no file exists, init dictionary
                    # where key=layer_idx, value=list of activations where each element is a minibatch of activations
                    saved_att = defaultdict(list)

                saved_att[i].append(padded_context)
                # Save to path in self.save_activation
                torch.save(saved_att, self.save_activation)

            if self.token_classifier_at is not None and self.token_classifier_at == i+1:
                output_dict['mid_layer_output'] = context.masked_fill(mask_src.permute(2, 0, 1), 0).type_as(context)
            # output of layer
        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        context = self.postprocess_layer(context)

        if self.add_adapter:
            context = self.adapters(context, input_lang)

        # Save final encoder output (after layer norm)
        if self.save_activation is not None:
            # Zero-pad layer output
            padded_context = context.masked_fill(mask_src.permute(2, 0, 1), 0).type_as(context)
            try:
                # Load existing file
                saved_att = torch.load(self.save_activation + '.norm')
            except OSError:
                # In case no file exists, init dictionary
                # where key=layer_idx, value=list of activations where each element is a minibatch of activations
                saved_att = defaultdict(list)
            # key=-1 to symbolize last encoder output (after layer norm)
            saved_att[-1].append(padded_context)
            torch.save(saved_att, self.save_activation + '.norm')

        output_dict['context'] = context
        output_dict['src_mask'] = mask_src
        # {'context': context, 'src_mask': mask_src}

        # return context, mask_src
        return output_dict


class TransformerDecoder(nn.Module):
    """Decoder in 'Attention is all you need'"""

    def __init__(self, opt, embedding, positional_encoder,
                 language_embeddings=None, ignore_source=False, allocate_positions=True):
        """
        :param opt:
        :param embedding:
        :param positional_encoder:
        :param attribute_embeddings:
        :param ignore_source:
        """
        super(TransformerDecoder, self).__init__()

        self.model_size = opt.model_size
        self.n_heads = opt.n_heads
        self.inner_size = opt.inner_size
        self.layers = opt.layers
        self.dropout = opt.dropout
        self.word_dropout = opt.word_dropout
        self.attn_dropout = opt.attn_dropout
        self.emb_dropout = opt.emb_dropout
        self.encoder_type = opt.encoder_type
        self.ignore_source = ignore_source
        self.encoder_cnn_downsampling = opt.cnn_downsampling
        self.variational_dropout = opt.variational_dropout
        self.switchout = opt.switchout
        self.death_rate = opt.death_rate
        self.time = opt.time
        self.use_language_embedding = opt.use_language_embedding
        self.language_embedding_type = opt.language_embedding_type

        if self.switchout > 0:
            self.word_dropout = 0

        self.time_transformer = positional_encoder

        self.preprocess_layer = PrePostProcessing(self.model_size, self.emb_dropout, sequence='d',
                                                  variational=self.variational_dropout)

        self.postprocess_layer = PrePostProcessing(self.model_size, 0, sequence='n')

        self.word_lut = embedding

        # Using feature embeddings in models
        self.language_embeddings = language_embeddings  # feat_lut

        if self.language_embedding_type == 'concat':
            self.projector = nn.Linear(opt.model_size * 2, opt.model_size)

        self.positional_encoder = positional_encoder

        if allocate_positions:
            if hasattr(self.positional_encoder, 'len_max'):
                len_max = self.positional_encoder.len_max
                mask = torch.ByteTensor(np.triu(np.ones((len_max, len_max)), k=1).astype('uint8'))
                self.register_buffer('mask', mask)

        self.layer_modules = nn.ModuleList()
        self.build_modules()

        if opt.freeze_decoder:
            self.requires_grad_(False)
            # for p in self.parameters():
                # p.requires_grad = False
                # print('Require grad in Transformer Decoder:', p.requires_grad)

    def build_modules(self):

        e_length = expected_length(self.layers, self.death_rate)
        print("* Transformer Decoder with Absolute Attention with %.2f expected layers" % e_length)

        for _l in range(self.layers):
            # linearly decay the death rate
            death_r = (_l + 1.0) / self.layers * self.death_rate

            block = DecoderLayer(self.n_heads, self.model_size,
                                 self.dropout, self.inner_size, self.attn_dropout,
                                 variational=self.variational_dropout, death_rate=death_r)

            self.layer_modules.append(block)

        # self.layer_modules = nn.ModuleList([DecoderLayer(self.n_heads, self.model_size,
        #                                                  self.dropout, self.inner_size,
        #                                                  self.attn_dropout, variational=self.variational_dropout,
        #                                                  ignore_source=self.ignore_source) for _ in range(self.layers)])

    def renew_buffer(self, new_len):

        self.positional_encoder.renew(new_len)
        mask = torch.ByteTensor(np.triu(np.ones((new_len+1, new_len+1)), k=1).astype('uint8'))
        self.register_buffer('mask', mask)

    def process_embedding(self, input, input_lang=None):

        # if self.switchout == 0:
        #     input_ = input
        # if self.switchout > 0 and self.training:
        #     vocab_size = self.word_lut.weight.size(0)
        #     input_ = switchout(input, vocab_size, self.switchout)
        # else:
        input_ = input

        emb = embedded_dropout(self.word_lut, input_, dropout=self.word_dropout if self.training else 0)
        if self.time == 'positional_encoding':
            emb = emb * math.sqrt(self.model_size)
        """ Adding positional encoding """
        emb = self.time_transformer(emb)

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(input_lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                # print(emb.shape, lang_emb.shape, lang_emb.unsqueeze(1).shape)
                emb = emb + lang_emb.unsqueeze(1)  # TODO: there was a bug that dim didn't match. Additive doesn't work for zero-shot
                lang_emb.unsqueeze(1).repeat(1, input.size(1))
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                bos_emb = lang_emb.expand_as(emb[:, 0, :])
                emb[:, 0, :] = bos_emb  # B, T, Dim

                lang_emb = lang_emb.unsqueeze(1).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))
            else:
                raise NotImplementedError
        return emb

    def forward(self, input, context, src, input_lang=None, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (to be transposed)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """

        """ Embedding: batch_size x len_tgt x d_model """

        emb = self.process_embedding(input, input_lang)

        if context is not None:
            if self.encoder_type == "audio":
                if not self.encoder_cnn_downsampling:
                    mask_src = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                    mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
            else:

                mask_src = src.data.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.eq(onmt.constants.PAD).byte().unsqueeze(1) + self.mask[:len_tgt, :len_tgt]
        mask_tgt = torch.gt(mask_tgt, 0)

        # an ugly hack to bypass torch 1.2 breaking changes
        if torch_version >= 1.2:
            mask_tgt = mask_tgt.bool()

        output = self.preprocess_layer(emb.transpose(0, 1).contiguous())

        for i, layer in enumerate(self.layer_modules):

            output, coverage, _ = layer(output, context, mask_tgt, mask_src)  # batch_size x len_src x d_model

        # From Google T2T
        # if normalization is done in layer_preprocess, then it should also be done
        # on the output, since the output can grow very large, being the sum of
        # a whole stack of unnormalized layer outputs.
        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None, {'hidden': output, 'coverage': coverage, 'context': context})

        # return output, None
        return output_dict

    def step(self, input, decoder_state, **kwargs):
        """
        Inputs Shapes:
            input: (Variable) batch_size x len_tgt (to be transposed)
            context: (Variable) batch_size x len_src x d_model
            mask_src (Tensor) batch_size x len_src
            buffer (List of tensors) List of batch_size * len_tgt-1 * d_model for self-attention recomputing
        Outputs Shapes:
            out: batch_size x len_tgt x d_model
            coverage: batch_size x len_tgt x len_src

        """
        context = decoder_state.context
        buffers = decoder_state.attention_buffers
        lang = decoder_state.tgt_lang
        mask_src = decoder_state.src_mask

        if decoder_state.concat_input_seq:
            if decoder_state.input_seq is None:
                decoder_state.input_seq = input
            else:
                # concatenate the last input to the previous input sequence
                decoder_state.input_seq = torch.cat([decoder_state.input_seq, input], 0)
            input = decoder_state.input_seq.transpose(0, 1)

            src = decoder_state.src.transpose(0, 1) if decoder_state.src is not None else None

        if input.size(1) > 1:
            input_ = input[:, -1].unsqueeze(1)
        else:
            input_ = input
        """ Embedding: batch_size x 1 x d_model """
        check = input_.gt(self.word_lut.num_embeddings)
        emb = self.word_lut(input_)

        """ Adding positional encoding """
        emb = emb * math.sqrt(self.model_size)
        emb = self.time_transformer(emb, t=input.size(1))
        # emb should be batch_size x 1 x dim

        if self.use_language_embedding:
            lang_emb = self.language_embeddings(lang)  # B x H or 1 x H
            if self.language_embedding_type == 'sum':
                emb = emb + lang_emb
            elif self.language_embedding_type == 'concat':
                # replace the bos embedding with the language
                if input.size(1) == 1:  # at 1st time step
                    bos_emb = lang_emb.expand_as(emb[:, 0, :])
                    emb[:, 0, :] = bos_emb

                lang_emb = lang_emb.unsqueeze(1).expand_as(emb)
                concat_emb = torch.cat([emb, lang_emb], dim=-1)
                emb = torch.relu(self.projector(concat_emb))

                # if self.enable_feature:
                #     input_attbs = input_attbs.unsqueeze(1)
                #     attb_emb = self.feat_lut(input_attbs)
                #
                #     emb = torch.cat([emb, attb_emb], dim=-1)
                #
                #     emb = torch.relu(self.feature_projector(emb))

            else:
                raise NotImplementedError

        emb = emb.transpose(0, 1)

        # batch_size x 1 x len_src
        if context is not None:
            if mask_src is None:
                if self.encoder_type == "audio":
                    if src.data.dim() == 3:
                        if self.encoder_cnn_downsampling:
                            long_mask = src.data.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD)
                            mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                        else:
                            mask_src = src.narrow(2, 0, 1).squeeze(2).eq(onmt.constants.PAD).unsqueeze(1)
                    elif self.encoder_cnn_downsampling:
                        long_mask = src.eq(onmt.constants.PAD)
                        mask_src = long_mask[:, 0:context.size(0) * 4:4].unsqueeze(1)
                    else:
                        mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
                else:
                    mask_src = src.eq(onmt.constants.PAD).unsqueeze(1)
        else:
            mask_src = None

        len_tgt = input.size(1)
        mask_tgt = input.eq(onmt.constants.PAD).byte().unsqueeze(1)
        mask_tgt = mask_tgt + self.mask[:len_tgt, :len_tgt].type_as(mask_tgt)
        mask_tgt = torch.gt(mask_tgt, 0)
        # only get the final step of the mask during decoding (because the input of the network is only the last step)
        mask_tgt = mask_tgt[:, -1, :].unsqueeze(1)

        if torch_version >= 1.2:
            mask_tgt = mask_tgt.bool()

        output = emb.contiguous()

        for i, layer in enumerate(self.layer_modules):
            buffer = buffers[i] if i in buffers else None
            assert (output.size(0) == 1)

            # output, coverage, buffer = layer.step(output, context, mask_tgt, mask_src, buffer=buffer)
            output, coverage, buffer = layer(output, context, mask_tgt, mask_src,
                                             incremental=True, incremental_cache=buffer)

            decoder_state.update_attention_buffer(buffer, i)

        output = self.postprocess_layer(output)

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['coverage'] = coverage
        output_dict['context'] = context

        return output_dict


class Transformer(NMTModel):
    """Main model in 'Attention is all you need' """

    def __init__(self, encoder, decoder, generator=None, mirror=False, ctc=None):
        super().__init__(encoder, decoder, generator, ctc)
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout
        self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)

        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None, reverse_src_tgt=False):
        """
        :param streaming_state:
        :param streaming:
        :param mirror: if using mirror network for future anticipation
        :param batch: data object sent from the dataset
        :param target_mask:
        :param zero_encoder: zero out the encoder output (if necessary)
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        if not reverse_src_tgt:
            src = batch.get('source')
            tgt = batch.get('target_input')
            src_pos = batch.get('source_pos')
            tgt_pos = batch.get('target_pos')
            src_lang = batch.get('source_lang')
            tgt_lang = batch.get('target_lang')
            src_lengths = batch.src_lengths
            tgt_lengths = batch.tgt_lengths
        else:   # use target as input, source as output (instead of using source as input, target as output)
            src = batch.get('target_as_source')
            tgt = batch.get('source_input')
            src_pos = batch.get('target_pos')
            tgt_pos = batch.get('source_pos')
            src_lang = batch.get('target_lang')
            tgt_lang = batch.get('source_lang')
            src_lengths = batch.tgt_lengths
            tgt_lengths = batch.src_lengths

        src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, streaming=streaming,
                                      src_lengths=src_lengths, streaming_state=streaming_state)

        encoder_output = defaultdict(lambda : None, encoder_output)
        context = encoder_output['context']

        # the state is changed
        streaming_state = encoder_output['streaming_state']

        # zero out the encoder part for pre-training
        if zero_encoder:
            context.zero_()

        if not self.ctc:    # if CTC is active, skip decoder (if any)
            decoder_output = self.decoder(tgt, context, src, input_lang=tgt_lang, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths, streaming_state=streaming_state)

            # update the streaming state again
            decoder_output = defaultdict(lambda: None, decoder_output)
            streaming_state = decoder_output['streaming_state']
            output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output if not self.ctc else None
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['streaming_state'] = streaming_state if not self.ctc else None

        # final layer: computing softmax
        if not self.ctc:
            logprobs = self.generator[0](output_dict)
            output_dict['logprobs'] = logprobs

        if self.encoder.language_classifier:  # make language classification
            logprobs_lan = self.generator[1](encoder_output)
            output_dict['logprobs_lan'] = logprobs_lan

        # Mirror network: reverse the target sequence and perform backward language model
        if mirror:
            tgt_reverse = torch.flip(batch.get('target_input'), (0, ))
            tgt_pos = torch.flip(batch.get('target_pos'), (0, ))
            tgt_reverse = tgt_reverse.transpose(0, 1)
            # perform an additional backward pass
            reverse_decoder_output = self.mirror_decoder(tgt_reverse, context, src, input_lang=tgt_lang, input_pos=tgt_pos)

            reverse_decoder_output['src'] = src
            reverse_decoder_output['context'] = context
            reverse_decoder_output['target_mask'] = target_mask
            reverse_logprobs = self.generator[0](reverse_decoder_output)

            output_dict['reverse_hidden'] = reverse_decoder_output['hidden']
            output_dict['reverse_logprobs'] = reverse_logprobs

            # learn weights for mapping (g in the paper)
            output_dict['hidden'] = self.mirror_g(output_dict['hidden'])

        # compute the logits for each encoder step
        if self.ctc:
            output_dict['encoder_logits'] = self.generator[0](output_dict, input_name='context')
            # context T x B x H, so is encoder_logits originally
            # permute encoder_logits to B x H x T, since upsampler works with
            # minibatch x channels x [optional depth] x [optional height] x width
            # encoder_logits = self.generator[0](output_dict, input_name='context').permute(1, 2, 0)
            # permute back to T x B x H
            # output_dict['encoder_logits'] = self.ctc_upsampler()
            #.permute(2, 0, 1)

        # # This step removes the padding to reduce the load for the final layer
        # if target_masking is not None:
        #     output = output.contiguous().view(-1, output.size(-1))
        #
        #     mask = target_masking
        #     """ We remove all positions with PAD """
        #     flattened_mask = mask.view(-1)
        #
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
        #
        #     output = output.index_select(0, non_pad_indices)

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src, input_pos=src_pos, input_lang=src_lang)['context']

        if hasattr(self, 'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()
        decoder_output = self.decoder(tgt_input, context, src, input_lang=tgt_lang, input_pos=tgt_pos)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder') and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_out)
            else:
                gen_t = self.generator(dec_out)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, input_t, decoder_state, streaming=False):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param streaming:
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """
        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        # squeeze to remove the time step dimension
        log_prob = self.generator[0](output_dict).squeeze(0)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        # output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_atb = batch.get('target_atb')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)

        encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], encoder_output['src_mask'],
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering)

        return decoder_state

    def init_stream(self):

        pass

    def set_memory_size(self, src_memory_size, tgt_memory_size):

        pass


class DoubleTransformer(NMTModel):
    """Double decoder"""

    def __init__(self, encoder, decoder, decoder_aux, generator=None, mirror=False):
        super().__init__(encoder, decoder, generator)
        self.decoder_aux = decoder_aux  # added
        self.model_size = self.decoder.model_size
        self.switchout = self.decoder.switchout
        self.tgt_vocab_size = self.decoder.word_lut.weight.size(0)

        if self.encoder.input_type == 'text':
            self.src_vocab_size = self.encoder.word_lut.weight.size(0)
        else:
            self.src_vocab_size = 0

        if mirror:
            self.mirror_decoder = copy.deepcopy(self.decoder)
            self.mirror_g = nn.Linear(decoder.model_size, decoder.model_size)

    def reset_states(self):
        return

    def forward(self, batch, target_mask=None, streaming=False, zero_encoder=False,
                mirror=False, streaming_state=None):
        """
        :param streaming_state:
        :param streaming:
        :param mirror: if using mirror network for future anticipation
        :param batch: data object sent from the dataset
        :param target_mask:
        :param zero_encoder: zero out the encoder output (if necessary)
        :return:
        """
        if self.switchout > 0 and self.training:
            batch.switchout(self.switchout, self.src_vocab_size, self.tgt_vocab_size)

        src = batch.get('source')
        tgt = batch.get('target_input')
        src_pos = batch.get('source_pos')
        tgt_pos = batch.get('target_pos')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')
        src_lengths = batch.src_lengths
        tgt_lengths = batch.tgt_lengths

        if isinstance(src, list):  # ugly workaround to handle multi-way source data
            for i, src_i in enumerate(src):
                src[i] = src_i.transpose(0, 1)
        else:
            src = src.transpose(0, 1)  # transpose to have batch first
        tgt = tgt.transpose(0, 1)

        encoder_output = self.encoder(src, input_pos=src_pos, input_lang=src_lang, streaming=streaming,
                                      src_lengths=src_lengths, streaming_state=streaming_state)

        encoder_output = defaultdict(lambda: None, encoder_output)
        context = encoder_output['context']
        context_aux = encoder_output['context_aux']

        # the state is changed
        streaming_state = encoder_output['streaming_state']

        # zero out the encoder part for pre-training
        if zero_encoder:
            context.zero_()

        if isinstance(src, list):  # ugly workaround to handle multi-way source data
            src = src[0]  # currently this takes main-lan data. will not be used in decoder (unless w/ audio input)
        # TODO: src and tgt length!
        decoder_output = self.decoder(tgt, context, src, input_lang=tgt_lang, input_pos=tgt_pos, streaming=streaming,
                                      src_lengths=src_lengths, tgt_lengths=tgt_lengths, streaming_state=streaming_state,
                                      src_mask=encoder_output['src_mask'])

        # update the streaming state again
        decoder_output = defaultdict(lambda: None, decoder_output)
        streaming_state = decoder_output['streaming_state']
        output = decoder_output['hidden']

        output_dict = defaultdict(lambda: None)
        output_dict['hidden'] = output
        output_dict['context'] = context
        output_dict['src_mask'] = encoder_output['src_mask']
        output_dict['src'] = src
        output_dict['target_mask'] = target_mask
        output_dict['streaming_state'] = streaming_state

        # final layer: computing softmax
        logprobs = self.generator[0](output_dict)
        output_dict['logprobs'] = logprobs

        if context_aux is not None:
            decoder_aux_output = self.decoder_aux(tgt, context_aux, src, input_lang=tgt_lang, input_pos=tgt_pos,
                                                  streaming=streaming,
                                                  src_lengths=src_lengths, tgt_lengths=tgt_lengths,
                                                  streaming_state=streaming_state,
                                                  src_mask=encoder_output['src_mask_aux'])
            # also do this for aux output
            # streaming_state_aux =
            output_aux = decoder_aux_output['hidden']

            # repeat for aux
            output_dict['hidden_aux'] = output_aux
            output_dict['streaming_state_aux'] = decoder_aux_output['streaming_state']
        # # added for multi source decoder: these are passed on as part of output to calculate enc dist
        # if isinstance(src, list):
            output_dict['context_main'] = encoder_output['context_main']
            output_dict['context_aux'] = encoder_output['context_aux']
            output_dict['src_mask_main'] = encoder_output['src_mask_main']
            output_dict['src_mask_aux'] = encoder_output['src_mask_aux']

            logprobs_aux = self.generator[0](output_dict, 'hidden_aux')
            output_dict['logprobs_aux'] = logprobs_aux

        # Mirror network: reverse the target sequence and perform backward language model
        if mirror:
            tgt_reverse = torch.flip(batch.get('target_input'), (0, ))
            tgt_pos = torch.flip(batch.get('target_pos'), (0, ))
            tgt_reverse = tgt_reverse.transpose(0, 1)
            # perform an additional backward pass
            reverse_decoder_output = self.mirror_decoder(tgt_reverse, context, src, input_lang=tgt_lang, input_pos=tgt_pos)

            reverse_decoder_output['src'] = src
            reverse_decoder_output['context'] = context
            reverse_decoder_output['target_mask'] = target_mask
            reverse_logprobs = self.generator[0](reverse_decoder_output)

            output_dict['reverse_hidden'] = reverse_decoder_output['hidden']
            output_dict['reverse_logprobs'] = reverse_logprobs

            # learn weights for mapping (g in the paper)
            output_dict['hidden'] = self.mirror_g(output_dict['hidden'])

        # # This step removes the padding to reduce the load for the final layer
        # if target_masking is not None:
        #     output = output.contiguous().view(-1, output.size(-1))
        #
        #     mask = target_masking
        #     """ We remove all positions with PAD """
        #     flattened_mask = mask.view(-1)
        #
        #     non_pad_indices = torch.nonzero(flattened_mask).squeeze(1)
        #
        #     output = output.index_select(0, non_pad_indices)

        return output_dict

    def decode(self, batch):
        """
        :param batch: (onmt.Dataset.Batch) an object containing tensors needed for training
        :return: gold_scores (torch.Tensor) log probs for each sentence
                 gold_words  (Int) the total number of non-padded tokens
                 allgold_scores (list of Tensors) log probs for each word in the sentence
        """

        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_input = batch.get('target_input')
        tgt_output = batch.get('target_output')
        tgt_pos = batch.get('target_pos')
        # tgt_atb = batch.get('target_atb')  # a dictionary of attributes
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        # transpose to have batch first
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        batch_size = tgt_input.size(0)

        context = self.encoder(src, input_pos=src_pos, input_lang=src_lang)['context']

        if hasattr(self, 'autoencoder') and self.autoencoder \
                and self.autoencoder.representation == "EncoderHiddenState":
            context = self.autoencoder.autocode(context)

        gold_scores = context.new(batch_size).zero_()
        gold_words = 0
        allgold_scores = list()
        decoder_output = self.decoder(tgt_input, context, src, input_lang=tgt_lang, input_pos=tgt_pos)['hidden']

        output = decoder_output

        if hasattr(self, 'autoencoder') and self.autoencoder and \
                self.autoencoder.representation == "DecoderHiddenState":
            output = self.autoencoder.autocode(output)

        for dec_t, tgt_t in zip(output, tgt_output):

            dec_out = defaultdict(lambda: None)
            dec_out['hidden'] = dec_t.unsqueeze(0)
            dec_out['src'] = src
            dec_out['context'] = context

            if isinstance(self.generator, nn.ModuleList):
                gen_t = self.generator[0](dec_out)
            else:
                gen_t = self.generator(dec_out)
            gen_t = gen_t.squeeze(0)
            tgt_t = tgt_t.unsqueeze(1)
            scores = gen_t.gather(1, tgt_t)
            scores.masked_fill_(tgt_t.eq(onmt.constants.PAD), 0)
            gold_scores += scores.squeeze(1).type_as(gold_scores)
            gold_words += tgt_t.ne(onmt.constants.PAD).sum().item()
            allgold_scores.append(scores.squeeze(1).type_as(gold_scores))

        return gold_words, gold_scores, allgold_scores

    def renew_buffer(self, new_len):
        self.decoder.renew_buffer(new_len)

    def step(self, input_t, decoder_state, streaming=False):
        """
        Decoding function:
        generate new decoder output based on the current input and current decoder state
        the decoder state is updated in the process
        :param streaming:
        :param input_t: the input word index at time t
        :param decoder_state: object DecoderState containing the buffers required for decoding
        :return: a dictionary containing: log-prob output and the attention coverage
        """

        output_dict = self.decoder.step(input_t, decoder_state, streaming=streaming)
        output_dict['src'] = decoder_state.src.transpose(0, 1)

        # squeeze to remove the time step dimension
        log_prob = self.generator[0](output_dict).squeeze(0)

        coverage = output_dict['coverage']
        last_coverage = coverage[:, -1, :].squeeze(1)

        # output_dict = defaultdict(lambda: None)

        output_dict['log_prob'] = log_prob
        output_dict['coverage'] = last_coverage

        return output_dict

    def create_decoder_state(self, batch, beam_size=1, type=1, buffering=True, **kwargs):
        """
        Generate a new decoder state based on the batch input
        :param streaming:
        :param type:
        :param batch: Batch object (may not contain target during decoding)
        :param beam_size: Size of beam used in beam search
        :return:
        """
        src = batch.get('source')
        src_pos = batch.get('source_pos')
        tgt_atb = batch.get('target_atb')
        src_lang = batch.get('source_lang')
        tgt_lang = batch.get('target_lang')

        src_transposed = src.transpose(0, 1)

        encoder_output = self.encoder(src_transposed, input_pos=src_pos, input_lang=src_lang)
        decoder_state = TransformerDecodingState(src, tgt_lang, encoder_output['context'], encoder_output['src_mask'],
                                                 beam_size=beam_size, model_size=self.model_size,
                                                 type=type, buffering=buffering)

        return decoder_state

    def init_stream(self):

        pass

    def set_memory_size(self, src_memory_size, tgt_memory_size):

        pass


class TransformerDecodingState(DecoderState):

    def __init__(self, src, tgt_lang, context, src_mask, beam_size=1, model_size=512, type=2,
                 cloning=True, buffering=False):

        self.beam_size = beam_size
        self.model_size = model_size
        self.attention_buffers = dict()
        self.buffering = buffering

        if type == 1:
            # if audio only take one dimension since only used for mask
            # raise NotImplementedError
            self.original_src = src  # TxBxC
            self.concat_input_seq = True

            if src is not None:
                if src.dim() == 3:
                    self.src = src.narrow(2, 0, 1).squeeze(2).repeat(1, beam_size)
                    print(self.src.size())
                    # self.src = src.repeat(1, beam_size, 1) # T x Bb x c
                else:
                    self.src = src.repeat(1, beam_size)
            else:
                self.src = None

            if context is not None:
                self.context = context.repeat(1, beam_size, 1)
            else:
                self.context = None

            self.input_seq = None
            self.src_mask = None
            self.tgt_lang = tgt_lang

            # if tgt_atb is not None:
            #     self.use_attribute = True
            #     self.tgt_atb = tgt_atb
            #     # self.tgt_atb = tgt_atb.repeat(beam_size)  # size: Bxb
            #     for i in self.tgt_atb:
            #         self.tgt_atb[i] = self.tgt_atb[i].repeat(beam_size)
            # else:
            #     self.tgt_atb = None

        elif type == 2:
            bsz = src.size(1)  # src is T x B
            new_order = torch.arange(bsz).view(-1, 1).repeat(1, self.beam_size).view(-1)
            new_order = new_order.to(src.device)

            if cloning:
                self.src = src.index_select(1, new_order)  # because src is batch first

                if context is not None:
                    self.context = context.index_select(1, new_order)
                else:
                    self.context = None

                if src_mask is not None:
                    self.src_mask = src_mask.index_select(0, new_order)
                else:
                    self.src_mask = None
            else:
                self.context = context
                self.src = src
                self.src_mask = src_mask

            self.concat_input_seq = False
            self.tgt_lang = tgt_lang

        else:
            raise NotImplementedError

    def update_attention_buffer(self, buffer, layer):

        self.attention_buffers[layer] = buffer  # dict of 2 keys (k, v) : T x B x H

    def update_beam(self, beam, b, remaining_sents, idx):

        if self.beam_size == 1:
            return

        for tensor in [self.src, self.input_seq]:

            if tensor is None:
                continue

            t_, br = tensor.size()
            sent_states = tensor.view(t_, self.beam_size, remaining_sents)[:, :, idx]

            sent_states.copy_(sent_states.index_select(
                1, beam[b].getCurrentOrigin()))

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            if buffer_ is None:
                continue

            for k in buffer_:
                t_, br_, d_ = buffer_[k].size()
                sent_states = buffer_[k].view(t_, self.beam_size, remaining_sents, d_)[:, :, idx, :]

                sent_states.data.copy_(sent_states.data.index_select(
                    1, beam[b].getCurrentOrigin()))

    # in this section, the sentences that are still active are
    # compacted so that the decoder is not run on completed sentences
    def prune_complete_beam(self, active_idx, remaining_sents):

        model_size = self.model_size

        def update_active_with_hidden(t):
            if t is None:
                return t
            dim = t.size(-1)
            # select only the remaining active sentences
            view = t.data.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            return view.index_select(1, active_idx).view(*new_size)

        def update_active_without_hidden(t):
            if t is None:
                return t
            view = t.view(-1, remaining_sents)
            new_size = list(t.size())
            new_size[-1] = new_size[-1] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            return new_t

        self.context = update_active_with_hidden(self.context)

        self.input_seq = update_active_without_hidden(self.input_seq)

        if self.src.dim() == 2:
            self.src = update_active_without_hidden(self.src)
        elif self.src.dim() == 3:
            t = self.src
            dim = t.size(-1)
            view = t.view(-1, remaining_sents, dim)
            new_size = list(t.size())
            new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
            new_t = view.index_select(1, active_idx).view(*new_size)
            self.src = new_t

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]

            for k in buffer_:
                buffer_[k] = update_active_with_hidden(buffer_[k])

    # For the new decoder version only
    def _reorder_incremental_state(self, reorder_state):

        if self.context is not None:
            self.context = self.context.index_select(1, reorder_state)

        if self.src_mask is not None:
            self.src_mask = self.src_mask.index_select(0, reorder_state)
        self.src = self.src.index_select(1, reorder_state)

        for l in self.attention_buffers:
            buffer_ = self.attention_buffers[l]
            if buffer_ is not None:
                for k in buffer_.keys():
                    t_, br_, d_ = buffer_[k].size()
                    buffer_[k] = buffer_[k].index_select(1, reorder_state)  # 1 for time first


