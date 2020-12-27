import time
import numpy as np
import pandas as pd
from bokeh import models as bom
from bokeh import plotting as bop
from bokeh import io as boi
from bokeh import layouts as bol
from contextlib import contextmanager


def array(fig):
    fig.canvas.draw_idle()
    renderer = fig.canvas.get_renderer()
    w, h = int(renderer.width), int(renderer.height)
    return (np.frombuffer(renderer.buffer_rgba(), np.uint8)
                        .reshape((h, w, 4))
                        [:, :, :3]
                        .copy())

class Stream:

    def __init__(self, run=-1, prefix=''):
        super().__init__()
        bop.output_notebook(hide_banner=True)

        self._reader = reading.Reader(run, prefix)

        self._source = bom.ColumnDataSource({'time_': np.array([0])})
        self._handle = None

    def _new_grid(self, children):
        return bol.gridplot(children, ncols=4, plot_width=350, plot_height=300, merge_tools=False)

    def _init(self, df):
        self._source = bom.ColumnDataSource(df.reset_index())

        children = []
        for (category, title), info in reading.sourceinfo(df).groupby(['category', 'title']):
            plotter = CATEGORIES[category]['plotter']
            f = plotter(self._source, info)
            f.title = bom.Title(text=title)
            children.append(f)
        self._grid = self._new_grid(children)

        ## TODO: Not wild about the flicker that comes from clearing this way
        from IPython.display import clear_output
        clear_output(wait=True)
        self._handle = bop.show(self._grid, notebook_handle=True)

    def update(self, rule='60s', df=None):
        # Drop the last row as it'll be constantly refreshed as the period occurs
        df = self._reader.resample(rule).iloc[:-1] if df is None else df
        info = reading.sourceinfo(df)

        has_new_cols = not info.id.isin(self._source.data).all()
        if has_new_cols:
            self._init(df)
        else:
            threshold = len(self._source.data['time_'])
            new = df.iloc[threshold:].reset_index()
            new.columns = ['_'.join(c) for c in new.columns]
            self._source.stream(new.reset_index())
        
        boi.push_notebook(handle=self._handle)

def view(run_name=-1, prefix='', rule='60s'):
    stream = Stream(run_name, prefix)
    while True:
        stream.update(rule=rule)
        time.sleep(1)

def review(run_name=-1, prefix='', rule='60s'):
    stream = Stream(run_name, prefix)
    stream.update(rule=rule)

def test_stream():
    times = pd.TimedeltaIndex([0, 60e3, 120e3])
    dfs = [
        pd.DataFrame([[0]], columns=['a'], index=times[:1]),
        pd.DataFrame([[0, 1], [10, 20]], columns=['a', 'b/a'], index=times[:2]),
        pd.DataFrame([[0, 1, 2], [10, 20, 30], [100, 200, 300]], columns=['a', 'b/a', 'b/b'], index=times[:3])]

    stream = Stream()
    for df in dfs:
        stream.update(df)
        time.sleep(1)
    