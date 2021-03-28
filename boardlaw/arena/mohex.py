import pandas as pd
from .. import sql, mohex, analysis
from . import common

def evaluate(snap_id, n_envs=2):
    row = sql.query('select * from snaps where id == ?', params=(snap_id,)).iloc[0]
    worlds = common.worlds(row.run, n_envs=n_envs)
    ags = {
        snap_id: common.agent(row.run, row.idx, device='cpu'),
        'mhx': mohex.MoHexAgent()}
    return common.evaluate(worlds, ags) 