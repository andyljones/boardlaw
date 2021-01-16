from pavlov.stats.timeseries.formatters import channel
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

class Plotter:

    def __init__(self, run=-1, prefix='', rule='60s'):
        self.run = run
        self.readers = registry.StatsReaders(run)
        self.groups = {}
        self.plotters = {}
        self.handle = None
        self.rule = rule

        bop.output_notebook(hide_banner=True)
        self.refresh()

    def refresh_groups(self):
        self.readers.refresh()
        reinit = False
        for prefix, reader in self.readers.items():
            s = registry.parse_prefix(prefix)
            key = (type(reader), s.group)
            if key not in self.groups:
                self.groups[key] = []
            if prefix not in self.groups[key]:
                self.groups[key].append(prefix)
                reinit = True
        return reinit

    def initialize(self, **kwargs):
        plotters = {}
        for subplot, prefixes in self.groups.items():
            readers = [self.readers[p] for p in prefixes] 
            plotters[subplot] = readers[0].plotter(readers, **kwargs)
        self.plotters = plotters

        children = [p.figure for p in self.plotters.values()]
        grid = bol.gridplot(children, ncols=4, plot_width=350, plot_height=300, merge_tools=False)

        from IPython.display import clear_output
        clear_output(wait=True)
        self.handle = bop.show(grid, notebook_handle=True)

    def refresh(self):
        reinit = self.refresh_groups()
        if reinit:
            self.initialize(rule=self.rule)

        for subplot, plotter in self.plotters.items():
            plotter.refresh()

        if self.handle:
            boi.push_notebook(handle=self.handle)

def review(run=-1, **kwargs):
    Plotter(run, **kwargs)

def view(run=-1, **kwargs):
    plotter = Plotter(run, **kwargs)
    while True:
        plotter.refresh()
        time.sleep(1.)

@tests.mock_dir
@tests.mock_time
def demo():
    from . import mean, mean_std, mean_percent
    run = runs.new_run()
    with registry.to_run(run):
        plotter = Plotter(run)
        plotter.refresh()
        time.sleep(1)
        tests.set_time(30)

        mean('single', 1)
        plotter.refresh()
        tests.set_time(90)
        time.sleep(1)

        mean('single', 2)
        plotter.refresh()
        tests.set_time(150)
        time.sleep(1)

        mean('double.one', 2)
        mean('double.two', 3)
        mean_std('ms', 1, 1)
        mean_percent('mp', 1, 1)
        plotter.refresh()
        tests.set_time(210)
        time.sleep(1)

        mean('single', 4)
        mean('double.one', 5)
        mean('double.two', 6)
        mean('new', 7)
        mean_std('ms', 1, 1)
        plotter.refresh()