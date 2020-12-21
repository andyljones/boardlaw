#/home/ajones/conda/envs/test/bin/python
# import torch
# torch.tensor(1).cuda()
# from boardlaw.mcts.cuda import *

# args = torch.load('output/tmp.pkl').to('cuda:1')
# m = mcts(**args)
# result = descend(m)
from boardlaw import arena
from boardlaw.main.common import *
arena.mohex_arena('2020-12-21 14-27-26 az-test', worldfunc, agentfunc, device='cuda:1')