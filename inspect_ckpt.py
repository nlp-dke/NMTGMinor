import torch
import sys
from onmt.Dict import Dict

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

print("***Key in ckpt:")
for k in ckpt:
    print(k)

print("***Key in model:")
for k in ckpt['model']:
    print(k, ckpt['model'][k].shape)

print("***Key in dicts:")
for k in ckpt['dicts']:
    print(k, ckpt['dicts'][k])
    if type(ckpt['dicts'][k]) is Dict:
        print('*** Size', ckpt['dicts'][k].size())
