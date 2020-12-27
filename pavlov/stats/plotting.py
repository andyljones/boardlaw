import aljpy
import re
import time
import numpy as np
import pandas as pd
from bokeh import models as bom
from bokeh import plotting as bop
from bokeh import io as boi
from bokeh import layouts as bol
from contextlib import contextmanager
from .. import runs, tests
from . import registry
from collections import defaultdict

def split(key):
    parts = key.split('.')
    if len(parts) == 2:
        return aljpy.dotdict(
            kind=parts[0],
            subplot='.'.join(parts), 
            title='.'.join(parts[1:]),
            label='')
    else:
        return aljpy.dotdict(
            kind=parts[0],
            subplot='.'.join(parts[:-1]), 
            title='.'.join(parts[1:-1]),
            label=parts[-1])


class Plotter:

    def __init__(self, run=-1, prefix='', rule='60s'):
        self.run = run
        self.pool = registry.ReaderPool(run)
        self.readers = {}
        self.plotters = {}
        self.handle = None
        self.rule = rule

        bop.output_notebook(hide_banner=True)
        self.refresh()

    def refresh_pool(self):
        self.pool.refresh()
        reinit = False
        for key, reader in self.pool.pool.items():
            s = split(key)
            if s.subplot not in self.readers:
                self.readers[s.subplot] = {}
            if key not in self.readers[s.subplot]:
                self.readers[s.subplot][key] = reader
                reinit = True
        return reinit

    def initialize(self, **kwargs):
        plotters = {}
        for subplot, readers in self.readers.items():
            prototype = list(readers.values())[0]
            plotters[subplot] = prototype.plotter(readers, **kwargs)
        self.plotters = plotters

        children = [p.figure for p in self.plotters.values()]
        grid = bol.gridplot(children, ncols=4, plot_width=350, plot_height=300, merge_tools=False)

        from IPython.display import clear_output
        clear_output(wait=True)
        self._handle = bop.show(grid, notebook_handle=True)

    def refresh(self):
        reinit = self.refresh_pool()
        if reinit:
            self.initialize(rule=self.rule)

        for subplot, plotter in self.plotters.items():
            plotter.refresh()

        boi.push_notebook(handle=self._handle)

def review(run=-1):
    Plotter(run)

def view(run=-1):
    plotter = Plotter(run)
    while True:
        plotter.refresh()
        time.sleep(1.)

@tests.mock_dir
@tests.mock_time
def demo():
    from . import mean, mean_std, mean_percent
    run = runs.new_run()
    with registry.to_run(run):
        tests.set_time(30)
        mean('single', 1)
        mean('double.one', 2)
        mean('double.two', 3)

        mean_std('ms', 1, 1)

        mean_percent('ms', 1, 1)

        plotter = Plotter(run)

        tests.set_time(90)
        time.sleep(1)
        mean('single', 4)
        mean('double.one', 5)
        mean('double.two', 6)
        mean('new', 7)

        mean_std('ms', 1, 1)
        plotter.refresh()