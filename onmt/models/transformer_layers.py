import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.utils.weight_norm as WeightNorm
import onmt 
import torch.nn.functional as F
from onmt.modules.bottle import Bottle
from onmt.modules.static_dropout import StaticDropout
from onmt.modules.linear import XavierLinear as Linear
from onmt.modules.linear import XavierLinear
from onmt.modules.linear import group_linear, FeedForwardSwish
from onmt.modules.linear import FeedForward
from onmt.modules.attention import MultiHeadAttention
from onmt.modules.dropout import VariationalDropout

class PrePostProcessing(nn.Module):
    """Applies processing to tensors
    Args:
        d_model: dimension of model
        p:       dropout probabolity  
        sequence of processing steps: 
            n = normalization
            d = dropout
            a = adding previous input to output (residual)
    """
    
    def __init__(self, d_model, dropout_p, sequence='nda', variational=False, elementwise_affine=True, death_rate=0.0):
        super(PrePostProcessing, self).__init__() 
        self.d_model = d_model
        self.dropout_p = dropout_p     
        
        self.steps = list(sequence)
        
        if onmt.constants.residual_type == 'gated':
            # gated residual
            # initialize k with one 
            self.k = nn.Parameter(torch.ones(1))
        
        if 'n' in self.steps:
            ln = nn.LayerNorm((self.d_model,),elementwise_affine=elementwise_affine)
            self.layer_norm = Bottle(ln)
        if 'd' in self.steps:
            if variational:
                self.dropout = VariationalDropout(self.dropout_p, batch_first=False)
            else:
                self.dropout = nn.Dropout(self.dropout_p)

        self.death_rate = death_rate
    
    def forward(self, tensor, input_tensor=None, mask=None):

        output = tensor
        for step in self.steps:
            if step == 'n':
                output = self.layer_norm(output, mask=mask)
                output = output
            if step == 'd':
                output = self.dropout(output)
            if step == 'a':
                if input_tensor is not None:
                    if onmt.constants.residual_type != 'gated':
                        output = output + input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor
            if step == 'b':
                if input_tensor is not None:
                    if onmt.constants.residual_type != 'gated':
                        output = output + input_tensor - input_tensor
                    else:
                        output = F.relu(self.k) * output + input_tensor - input_tensor
            if step == 'm':  # keep residual but, use mean pool instead
                if input_tensor is not None:
                    mask_ = mask.permute(2, 0, 1)  # B x H x T --> T x B x H
                    meanpool_tensor = torch.sum(input_tensor.float().masked_fill(mask_, 0).type_as(input_tensor), dim=0,
                                               keepdim=True) / (1 - mask_.float()).sum(dim=0)
                    # masked_fill_ is inplace. currently not inplace
                    if onmt.constants.residual_type != 'gated':
                        output = output + meanpool_tensor
                    else:
                        output = F.relu(self.k) * output + meanpool_tensor

            if step == 's':
                # stochastically drop residual by (layer_idx)/total_layers chance
                # bottom layer --> 0; top layer --> 1 - (1-total_layers)/total_layers
                if input_tensor is not None:
                    coin = True     # if coin is True, keep residual layer
                    if self.training:
                        coin = (torch.rand(1)[0].item() >= self.death_rate)

                    if self.training:
                        if self.death_rate == 1:
                            scaling_factor = 0
                        else:
                            scaling_factor = 1.0 / (1 - self.death_rate)
                    else:
                        scaling_factor = 1.0

                    if onmt.constants.residual_type != 'gated':
                        if coin:
                            output = output + input_tensor * scaling_factor
                        else:
                            output = output + input_tensor - input_tensor
                    else:
                        if coin:
                            output = F.relu(self.k) * output + input_tensor * scaling_factor
                        else:
                            output = F.relu(self.k) * output + input_tensor - input_tensor

        return output


class EncoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one encoder layer
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead:    multi-head attentions layer
        feedforward:  feed forward layer
    
    Input Shapes:
        query: batch_size x len_query x d_model 
        key:   batch_size x len_key x d_model   
        value: batch_size x len_key x d_model
        mask:  batch_size x len_query x len_key or broadcastable 
    
    Output Shapes:
        out: batch_size x len_query x d_model
    """
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, variational=False, death_rate=0.0,
                 change_residual=None, change_att_query=None, add_adapter=False, opt=None, residual_death_rate=0.0, **kwargs):
        super(EncoderLayer, self).__init__()
        self.variational = variational
        self.death_rate = death_rate

        if change_residual is None:
            att_postpro_type = 'da'  # dropout, normal residual
        else:
            if change_residual == 1:
                att_postpro_type = 'dm'  # dropout, no normal residual, use meanpool instead
            elif change_residual == 2:
                att_postpro_type = 'db'  # only dropout
            elif change_residual == 3:
                att_postpro_type = 'ds'    # stochastically drop residual
            else:
                raise NotImplementedError

        print('*** att_postpro_type', att_postpro_type, ', change_att_query', change_att_query)

        if opt.language_specific_encoder:
            # for language in
            ff_dict = {}

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence=att_postpro_type, variational=self.variational,
                                                  death_rate=residual_death_rate)
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
        self.multihead = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)

        if onmt.constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
        elif onmt.constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p, variational=self.variational)
        else:
            raise NotImplementedError
        self.feedforward = Bottle(feedforward)

        self.change_att_query = change_att_query

        self.add_adapter = add_adapter

        if self.add_adapter:
            adapter_bottleneck_size = opt.model_size // 2
            from onmt.modules.multilingual_factorized.multilingual_adapters import MultilingualAdapter
            self.adapters = MultilingualAdapter(model_size=opt.model_size, bottleneck_size=adapter_bottleneck_size,
                                                n_languages=opt.n_languages,
                                                dropout=opt.dropout, variational=self.variational,
                                                death_rate=opt.adapter_death_rate)
            
    def forward(self, input, attn_mask, given_query=None, att_plot_path=None,
                src_lang=None, incremental=False, incremental_cache=None, mems=None):

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
            query = self.preprocess_attn(input)

            if self.change_att_query is None:
                out, _, _ = self.multihead(query, query, query, attn_mask, att_plot_path=att_plot_path)
            else:
                out, _, _ = self.multihead(given_query, query, query, attn_mask, att_plot_path=att_plot_path)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input, mask=attn_mask)

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)

            if self.add_adapter:
                input = self.adapters(input, src_lang)
        
        return input
    

class DecoderLayer(nn.Module):
    """Wraps multi-head attentions and position-wise feed forward into one layer of decoder
    
    Args:
        h:       number of heads
        d_model: dimension of model
        p:       dropout probabolity 
        d_ff:    dimension of feed forward
        
    Params:
        multihead_tgt:  multi-head self attentions layer
        multihead_src:  multi-head encoder-decoder attentions layer        
        feedforward:    feed forward layer
    
    Input Shapes:
        query:    batch_size x len_query x d_model 
        key:      batch_size x len_key x d_model   
        value:    batch_size x len_key x d_model
        context:  batch_size x len_src x d_model
        mask_tgt: batch_size x len_query x len_key or broadcastable 
        mask_src: batch_size x len_query x len_src or broadcastable 
    
    Output Shapes:
        out:      batch_size x len_query x d_model
        coverage: batch_size x len_query x len_key
        
    """    
    
    def __init__(self, h, d_model, p, d_ff, attn_p=0.1, version=1.0, ignore_source=False,
                 variational=False, death_rate=0.0):
        super(DecoderLayer, self).__init__()
        self.version = version
        self.ignore_source = ignore_source
        self.variational = variational
        self.death_rate = death_rate

        self.preprocess_attn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        if not self.ignore_source:
            self.preprocess_src_attn = PrePostProcessing(d_model, p, sequence='n')
            self.postprocess_src_attn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)
            self.multihead_src = MultiHeadAttention(h, d_model, attn_p=attn_p, share=2)
        
        self.preprocess_ffn = PrePostProcessing(d_model, p, sequence='n')
        self.postprocess_ffn = PrePostProcessing(d_model, p, sequence='da', variational=self.variational)

        self.multihead_tgt = MultiHeadAttention(h, d_model, attn_p=attn_p, share=1)

        if onmt.constants.activation_layer == 'linear_relu_linear':
            ff_p = p
            feedforward = FeedForward(d_model, d_ff, ff_p, variational=self.variational)
        elif onmt.constants.activation_layer == 'maxout':
            k = int(math.ceil(d_ff / d_model))
            feedforward = MaxOut(d_model, d_model, k)
        elif onmt.constants.activation_layer == 'linear_swish_linear':
            ff_p = p
            feedforward = FeedForwardSwish(d_model, d_ff, ff_p)
        else:
            raise NotImplementedError
        self.feedforward = Bottle(feedforward)
    
    def forward(self, input, context, mask_tgt, mask_src,
                incremental=False, incremental_cache=None, reuse_source=True):
        
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """

        coverage = None

        coin = True
        if self.training:
            coin = (torch.rand(1)[0].item() >= self.death_rate)

        if coin:
        
            query = self.preprocess_attn(input)

            self_context = query

            out, _, incremental_cache = self.multihead_tgt(query, self_context, self_context, mask_tgt,
                                           incremental=incremental, incremental_cache=incremental_cache)

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_attn(out, input)

            """ Context Attention layer 
                layernorm > attn > dropout > residual
            """
            if not self.ignore_source:
                query = self.preprocess_src_attn(input)
                out, coverage, incremental_cache = self.multihead_src(query, context, context, mask_src,
                                                      incremental=incremental, incremental_cache=incremental_cache)

                if self.training and self.death_rate > 0:
                    out = out / (1 - self.death_rate)

                input = self.postprocess_src_attn(out, input)
            else:
                coverage = None

            """ Feed forward layer 
                layernorm > ffn > dropout > residual
            """
            out = self.feedforward(self.preprocess_ffn(input))

            if self.training and self.death_rate > 0:
                out = out / (1 - self.death_rate)

            input = self.postprocess_ffn(out, input)
    
        return input, coverage, incremental_cache
        
    def step(self, input, context, mask_tgt, mask_src, buffer=None):
        """ Self attention layer 
            layernorm > attn > dropout > residual
        """
        
        query = self.preprocess_attn(input)
        
        out, _, buffer = self.multihead_tgt.step(query, query, query, mask_tgt, buffer=buffer)

        input = self.postprocess_attn(out, input)
        
        """ Context Attention layer 
            layernorm > attn > dropout > residual
        """
        if not self.ignore_source:
            query = self.preprocess_src_attn(input)
            out, coverage, buffer = self.multihead_src.step(query, context, context, mask_src, buffer=buffer)
            input = self.postprocess_src_attn(out, input)
        else:
            coverage = None
        
        """ Feed forward layer 
            layernorm > ffn > dropout > residual
        """
        out = self.feedforward(self.preprocess_ffn(input))

        input = self.postprocess_ffn(out, input)
        
        return input, coverage, buffer


class PositionalEncoding(nn.Module):
    """Adds positional embeddings to standard word embeddings 
    This matches the original TensorFlow implementation at https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py.
    
    Args:
        d_model: dimension of model
        p:       dropout probability  
        len_max: max seq length for pre-calculated positional embeddings
        
    Inputs Shapes: 
        word_emb: batch_size x len_seq x d_model 
        
    Outputs Shapes:
        out:   batch_size x len_seq x d_model
        
    """

    def __init__(self, d_model, p=0, len_max=512, fixed_encoding=True):
        # save a fixed positional embedding matrix up to len_max,
        # so that no need to recreate it everytime
        super(PositionalEncoding, self).__init__()
        self.len_max = len_max
        self.d_model = d_model
        self.data_type = None
        self.fixed_encoding = fixed_encoding

        if fixed_encoding:
            self.renew(len_max)
        else:
            self.pos_emb = nn.Embedding(num_embeddings=self.len_max, embedding_dim=self.d_model)

        self.p = p

    def renew(self, new_max_len):

        if self.fixed_encoding:
            # detele the old variable to avoid Pytorch's error when register new buffer
            cuda = False
            if hasattr(self, 'pos_emb'):
                cuda = self.pos_emb.is_cuda
                # self.data_type = torch.type(self.pos_emb)
                del self.pos_emb

            position = torch.arange(0, new_max_len).float()

            num_timescales = self.d_model // 2
            log_timescale_increment = math.log(10000) / (num_timescales - 1)
            inv_timescales = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment)
            scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
            pos_emb = torch.cat((torch.sin(scaled_time), torch.cos(scaled_time)), 1)

            if cuda:
                pos_emb = pos_emb.cuda()

            if self.data_type is not None:
                pos_emb.type(self.data_type)
            # wrap in a buffer so that model can be moved to GPU
            self.register_buffer('pos_emb', pos_emb)
            # self.data_type = self.pos_emb.type()
            self.len_max = new_max_len

            # added with shorter wave length
            log_timescale_increment_ = math.log(10) / (num_timescales - 1)
            inv_timescales_ = torch.exp(torch.arange(0, num_timescales).float() * -log_timescale_increment_)
            scaled_time_ = position.unsqueeze(1) * inv_timescales_.unsqueeze(0)
            pos_emb_short = torch.cat((torch.sin(scaled_time_), torch.cos(scaled_time_)), 1)

            if cuda:
                pos_emb_short = pos_emb_short.cuda()

            if self.data_type is not None:
                pos_emb_short.type(self.data_type)
            # wrap in a buffer so that model can be moved to GPU
            self.register_buffer('pos_emb_short', pos_emb_short)

        else:   # Don't renew learned embedding.
            pass

    def forward(self, word_emb, t=None, change_code=None):

        len_seq = t if t else word_emb.size(1)

        self.data_type = word_emb.type()

        if self.fixed_encoding:
            if len_seq > self.len_max:
                self.renew(len_seq)

            if change_code == 2:
                if word_emb.size(1) == len_seq:
                    time_ = self.pos_emb[-len_seq:, :].type_as(word_emb)
                    out = word_emb + time_
                else:
                    time_emb = self.pos_emb[-(len_seq - 1), :]  # 1 x dim
                    out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1).type_as(word_emb)

            elif change_code == 3:
                if word_emb.size(1) == len_seq:
                    time_ = self.pos_emb_short[:len_seq, :].type_as(word_emb)
                    out = word_emb + time_
                else:
                    # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
                    time_emb = self.pos_emb_short[len_seq - 1, :]  # 1 x dim
                    # out should have size bs x 1 x dim
                    out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1).type_as(word_emb)

            else:
                if word_emb.size(1) == len_seq:
                    time_ = self.pos_emb[:len_seq, :].type_as(word_emb)
                    out = word_emb + time_
                else:
                    # out = word_emb + Variable(self.pos_emb[:len_seq, :][-1, :], requires_grad=False)
                    time_emb = self.pos_emb[len_seq - 1, :]  # 1 x dim
                    # out should have size bs x 1 x dim
                    out = word_emb + time_emb.unsqueeze(0).repeat(word_emb.size(0), 1, 1).type_as(word_emb)

            if change_code is not None:
                out -= word_emb

        else:
            if len_seq > self.len_max:
                raise ValueError('Input seq exceeds embedding size')

            if word_emb.size(1) == len_seq:
                pos_idx = torch.arange(len_seq).to(word_emb.device)
                out = word_emb + self.pos_emb(pos_idx)
            else:   # at decoding time, only need to add pos_emb on one position
                pos_idx = torch.arange(len_seq-1, len_seq).to(word_emb.device)
                out = word_emb + self.pos_emb(pos_idx).unsqueeze(0).repeat(word_emb.size(0), 1, 1).type_as(word_emb)

        return out
