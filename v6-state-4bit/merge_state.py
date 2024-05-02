from collections import OrderedDict
import os
import sys
from typing import Dict
import typing
import torch

if '-h' in sys.argv or '--help' in sys.argv:
    print(f'Usage: python3 {sys.argv[0]} <base_model.pth> <state_checkpoint.pth> <output.pth>')

base_model, state, output = sys.argv[1], sys.argv[2], sys.argv[3]

with torch.no_grad():
    w: Dict[str, torch.Tensor] = torch.load(base_model, map_location='cpu')
    # merge state-only slim checkpoint into the main weights
    w_state: Dict[str, torch.Tensor] = torch.load(state, map_location='cpu')
    for k in w_state.keys():
        w[k] = w_state[k]
    output_w: typing.OrderedDict[str, torch.Tensor] = OrderedDict()
    # merge state weights
    keys = list(w.keys())
    for k in keys:
        print(f'retaining {k}')
        output_w[k] = w[k].clone()
        del w[k]

    torch.save(output_w, output)
