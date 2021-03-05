# Implementation of the multilingual adapter as in Bapna et. al 2019

import torch
from torch.nn import Parameter
import torch.nn.functional as F
import math
# Commented out since the new components are not available
# from ..optimized.feed_forward import PositionWiseFeedForward
# from ..layer_norm import LayerNorm
import torch.nn as nn
from onmt.modules.linear import FeedForward

def xavier_normal(weight, gain=1.0):

    fan_in, fan_out = weight.size(-2), weight.size(-1)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    with torch.no_grad():
        weight.normal_(0, std)


class MultilingualAdapter(torch.nn.Module):

    def __init__(self, model_size, bottleneck_size, n_languages=1, dropout=0.0,
                 elementwise_affine=True, variational=False, death_rate=0.0):
        
        super(MultilingualAdapter, self).__init__()

        self.all_modules = torch.nn.ModuleList()

        for i in range(n_languages):
            layer_norm = nn.LayerNorm((model_size,), elementwise_affine=elementwise_affine)
            feed_forward = FeedForward(model_size, bottleneck_size, p=dropout, variational=variational)
            adapter = torch.nn.Sequential(layer_norm, feed_forward)
            self.all_modules.append(adapter)

        self.death_rate = death_rate

    def forward(self, input, lang=None):
        """
        :param input: TxBxN Tensor
        :param lang:  [1] Tensor
        :return:
        """
        adapter_lives = True

        if self.death_rate > 0:
            if self.training:
                adapter_lives = (torch.rand(1)[0].item() >= self.death_rate)
            if not adapter_lives:
                return input    # if rand < death rate, direclty give out input (adapter is dead)

        unique_lang_id = torch.unique(lang)
        assert len(unique_lang_id) == 1
        # Danni: This line was lang.numel() == 1. Had to change this after giving language tag to tokens:

        index = unique_lang_id.item()   # was lang.item()
        adapter = self.all_modules[index]

        # normalize -> transform -> residual
        return input + adapter(input)


