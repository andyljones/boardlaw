import os 
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

try:
    torch.cuda.init()
except:
    pass

torch.cuda.init()