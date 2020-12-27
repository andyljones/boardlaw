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
            match = re.fullmatch(r'(?P<subplot>.*)\.(?P<label>.*)', key)
            subplot = match.group('subplot')
            if subplot not in self.readers:
                self.readers[subplot] = {}
            if key not in self.readers[subplot]:
                self.readers[subplot][key] = reader
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


@tests.mock_dir
@tests.mock_time
def demo():
    from . import mean
    run = runs.new_run()
    with registry.to_run(run):
        tests.set_time(30)
        mean('test', 1)
        tests.set_time(90)
        mean('test', 2)
    
    plotter = Plotter(run)