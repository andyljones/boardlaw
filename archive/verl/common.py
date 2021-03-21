import numpy as np
import torch

class TestEnd(Exception):
    pass

def numpyify(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x