# Scaling Scaling Laws with Board Games
[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://andyljones.com/boardlaw)
[![Discord](https://img.shields.io/discord/765294874832273419)](https://discord.gg/xhfNqQv)

This is the code for our paper, [*Scaling Scaling Laws with Board Games*](https://arxiv.org/abs/2104.03113). 

[Documentation's here](https://andyljones.com/boardlaw).

# Bugs
There is a bug in the calculation of the lambda constant, see Issue #15. This likely affects the results in the paper and is the cause of my finding much different c\_puct than expected from the literature. I've left the bug intact so that this codebase generates results consistent with the paper, but if you derive another codebase from this one it's something you should fix.
