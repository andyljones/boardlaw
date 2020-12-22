# import torch
# torch.tensor(1).cuda()

# from boardlaw.mcts.cuda import *
# from rebar import arrdict

# args = torch.load('output/tmp.pkl').to('cuda:1')
# for t in range(1, 256):
#     stacked = arrdict.cat([args for _ in range(t)])
#     m = mcts(**stacked)
#     result = descend(m)

from boardlaw import arena
from boardlaw.main.common import *
arena.mohex_arena('2020-12-21 14-27-26 az-test', worldfunc, agentfunc, device='cuda:1')