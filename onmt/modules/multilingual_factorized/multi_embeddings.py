import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiEmbedding(torch.nn.Module):

    def __init__(self, vocab_sizes, emb_size, padding_idx=None):
        """
        :param vocab_sizes: dict from idx to vocab size
        :param emb_size: integer
        :param padding_idx: integer
        """

        # The problem here is that: this module requires different vocab size
        # And I do not know how to map from index to vocab size ...
        # (because Idk which index corresponds to which language)
        super(MultiEmbedding, self).__init__()
        self.embeddings = nn.ModuleDict()
        self.vocab_sizes = vocab_sizes

        for idx in vocab_sizes:
            vocab_size = vocab_sizes[idx]
            if vocab_size > 0:
                embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
            else:
                # just in case there is some language unused with vocab size 0 ...
                embedding = torch.nn.Identity()
            # self.embeddings.append(embedding)
            self.embeddings[str(idx)] = embedding

    def forward(self, input, lang=None):
        """
        :param input: int tensor [T x B] or [B x T]
        :param lang: unit tensor
        :return:
        """

        assert lang.numel() == 1

        index = lang.item()

        return self.embeddings[str(index)](input)


class MultiGenerator(nn.Module):

    def __init__(self, hidden_size, vocab_sizes, fix_norm=False):

        super(MultiGenerator, self).__init__()
        self.hidden_size = hidden_size
        # self.output_size = output_size

        self.linears = nn.ModuleDict()
        for idx in vocab_sizes:
            vocab_size = vocab_sizes[idx]
            output_size = vocab_size

            if output_size > 0:
                linear = nn.Linear(hidden_size, output_size)
                std_ = hidden_size ** -0.5

                nn.init.normal_(linear.weight, 0.0, std_)
            else:
                linear = torch.nn.Identity()

            self.linears[str(idx)] = linear

        self.fix_norm = fix_norm

    def forward(self, output_dicts, lang):
        """
        :param lang:
        :param output_dicts: dictionary contains the outputs from the decoder
        :return: logits (the elements before softmax)
        """
        assert lang.numel() == 1
        index = lang.item()
        linear = self.linears[str(index)]

        input = output_dicts['hidden']
        fix_norm = self.fix_norm
        target_mask = output_dicts['target_mask']

        if not fix_norm:
            logits = linear(input)
        else:
            normalized_weights = F.normalize(linear.weight, dim=-1)
            normalized_bias = linear.bias
            logits = F.linear(input, normalized_weights, normalized_bias)

        # softmax will be done at the loss function
        # output = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        output_dicts['logits'] = logits
        return output_dicts


class MultiLinear(nn.Module):

    def __init__(self, hidden_size, vocab_sizes):

        super(MultiLinear, self).__init__()
        self.hidden_size = hidden_size

        self.linears = nn.ModuleList()
        for idx in vocab_sizes:
            vocab_size = vocab_sizes[idx]
            output_size = vocab_size

            if output_size > 0:
                linear = nn.Linear(hidden_size, output_size)
                std_ = hidden_size ** -0.5

                nn.init.normal_(linear.weight, 0.0, std_)
            else:
                linear = torch.nn.Identity()

            self.linears.append(linear)

    def forward(self, input, lang):

        assert lang.numel() == 1
        index = lang.item()
        linear = self.linears[index]

        return linear(input)