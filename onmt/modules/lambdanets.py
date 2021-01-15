import torch
from torch import nn, einsum
from einops import rearrange


# helpers functions

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_rel_pos(n):
    pos = torch.meshgrid(torch.arange(n), torch.arange(n))
    pos = rearrange(torch.stack(pos), 'n i j -> (i j) n')  # [n*n, 2] pos[n] = (i, j)
    rel_pos = pos[None, :] - pos[:, None]                  # [n*n, n*n, 2] rel_pos[n, m] = (rel_i, rel_j)
    rel_pos += n - 1                                       # shift value range from [-n+1, n-1] to [0, 2n-2]
    return rel_pos


class LambdaLayer(nn.Module):

    def __init__(self, model_size, dim_k, n=None, r=None, heads=4, dim_u=1):

        super(LambdaLayer, self).__init__()

