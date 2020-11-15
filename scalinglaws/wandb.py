import logging
import os
import wandb

os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_DIR'] = './.wandb'

def init(**kwargs):
    assert 'name' in kwargs
    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    wandb.init(**{'project': 'scalinglaws', 'dir': './.wandb'}, **kwargs)

def mean(key, num, denom):
    pass