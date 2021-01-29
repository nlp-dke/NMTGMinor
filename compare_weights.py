import torch
import sys
from onmt.Dict import Dict
import logging

ckpt_path_1 = sys.argv[1]
ckpt_path_2 = sys.argv[2]
ckpt_1 = torch.load(ckpt_path_1, map_location=lambda storage, loc: storage)
ckpt_2 = torch.load(ckpt_path_2, map_location=lambda storage, loc: storage)

print("***Key in ckpt:")
for k in ckpt_1:
    try:
        print(k, ckpt_1[k] == ckpt_2[k])
        if k == 'dicts':
            print(len(ckpt_1[k]['src'].idxToLabel))
            print(len(ckpt_2[k]['tgt'].idxToLabel))
    except Exception as e:
        print('Not matching', repr(e))

print("***Key in model:")
for k in ckpt_1['model']:
    print(k, ckpt_1['model'][k].shape)
    try:
        print(torch.all(ckpt_1['model'][k] == ckpt_2['model'][k]))
    except Exception as e:
        print('Not matching', repr(e))
