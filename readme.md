This is the repo for my ongoing research into [scaling laws in multiplayer games](https://docs.google.com/document/d/1OwniAl1ocnqKHc4jtPVJzemm46q6ZgPVhXhmL2ZDIJw/edit), as supported by the [Survival and Flourishing Fund](http://survivalandflourishing.org/). It's open because there are some components that might be independently 
interesting to others.

Do *not* try to build directly upon this repo; it's under massive ongoing change. Nothing is stable. On a similar note, I'm happy to answer specific questions or chat about implementation choices - you can find me on the [RL Discord](https://discord.gg/xhfNqQv) most of the time - but you cant expect robustness or support. 

## AlphaZero Implementation
Part of this project is writing a fast, low-resource AlphaZero implementation for small board games. Right now it can
solve 9x9 Hex to perfect play in ~4 hours on a single RTX 2080 Ti.

<p align="center">
    <img src="boardlaw-scaling.png"/>
</p>

(FWIW this stalls out at 11x11, and figuring out why is on my to-do list)

'Perfect play' is judged by 'being on-par with [MoHex](https://github.com/cgao3/benzene-vanilla-cmake)', which claims perfect play on boards up to size 9x9.

Because of the low-resource constraint, this implementation does a few things unusually:

* The game - Hex - is vectorized and stepped [entirely on the GPU](boardlaw/hex/cpp/kernels.cu). Playing random actions, the 
throughput is ~100m boards/second on a 16k vectorization. This is high enough that it forms a negligible part of the 
run time, even with very small networks.
* The MCTS is vectorized and carried out [entirely on the GPU](boardlaw/mcts/cpp/kernels.cu) too.
* It leverages [Monte-Carlo Tree Search as Regularized Policy Optimization](https://arxiv.org/abs/2007.12509) to get away with doing ~64 sims compared to the usual ~800. It also subs out the bisection search recommended in the paper for a Newton solver, which is much faster.
* It disposes of the replay buffer since stale, repeated samples are bad for training speed ([p55, 57](https://arxiv.org/pdf/1912.06680.pdf))
* To suppress the cyclic behaviour that can show up in self-play without a replay buffer, for 20% of games it uses a league based on [OpenAI Five's (p59)](https://arxiv.org/pdf/1912.06680.pdf).
* It uses fully-connected resnets, as convnets seem to be overkill for boards this small.
* It uses [ReZero](https://arxiv.org/abs/2003.04887) initialization to skip out on (slow, annoying) layernorms and batchnorms.
* It uses a very small `c_puct` of 1/16. This turned out to be unexpectedly critical; I don't know whether it's a consequence of the MCTS-as-regularized-tree-search, or if it reflects some other mistaken calculation elsewhere in my implementation.

There are intentionally no game-specific features: this is all intended as a tool for exploring the power of generic machine learning systems.

A lot of this is un-ablated as of mid-Jan, so take it with a pinch of salt when deciding where to attribute performance to. 

### Running it
Anyway, to run whatever I've got it set to right this second,
```python
from boardlaw.main import *
run()
```
Once it's declared the run started, you can watch its progress from a second Jupyter instance with
```python
from pavlov import *
monitor(-1)
```
or 
```python
from pavlov import *
stats.review(-1)
```
for charts.

There's a [Docker image](docker) if you're having trouble with dependencies.

## ActiveElo
One frustration in writing this was in figuring out what pairs of agents should play against eachother to most rapidly nail down the Elo of a new agent. I eventually cracked and wrote [activelo](activelo) which uses a variational Bayes approach to suggest, based on the games played so far, which pair should be played next. It's built using the superb [geotorch](https://github.com/Lezcano/geotorch) constrained optimization toolkit.

## Citations
```
@software{boardlaw,
author = {{Andy L Jones}},
title = {boardlaw},
url = {https://www.github.com/andyljones/boardlaw},
version = {0.0},
date = {2021-01-20},
}
```