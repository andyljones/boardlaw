import re
import pandas as pd
from bokeh import models as bom
from bokeh import plotting as bop
from bokeh import io as boi
from bokeh import layouts as bol
from bokeh import events as boe

from bokeh.palettes import Category10_10
from itertools import cycle

from pandas.core.series import Series
from ..plotting import split

def timedelta_xaxis(f):
    f.xaxis.ticker = bom.tickers.DatetimeTicker()
    f.xaxis.formatter = bom.FuncTickFormatter(code="""
        // TODO: Add support for millis

        // Calculate the hours, mins and seconds
        var s = Math.floor(tick / 1e3);
        
        var m = Math.floor(s/60);
        var s = s - 60*m;
        
        var h = Math.floor(m/60);
        var m = m - 60*h;
        
        var h = h.toString();
        var m = m.toString();
        var s = s.toString();
        var pm = m.padStart(2, "0");
        var ps = s.padStart(2, "0");

        // Figure out what the min resolution is going to be
        var min_diff = Infinity;
        for (var i = 0; i < ticks.length-1; i++) {
            min_diff = Math.min(min_diff, ticks[i+1]-ticks[i]);
        }

        if (min_diff <= 60e3) {
            var min_res = 2;
        } else if (min_diff <= 3600e3) {
            var min_res = 1;
        } else {
            var min_res = 0;
        }

        // Figure out what the max resolution is going to be
        if (ticks.length > 1) {
            var max_diff = ticks[ticks.length-1] - ticks[0];
        } else {
            var max_diff = Infinity;
        }

        if (max_diff >= 3600e3) {
            var max_res = 0;
        } else if (max_diff >= 60e3) {
            var max_res = 1;
        } else {
            var max_res = 2;
        }

        // Format the timedelta. Finally.
        if ((max_res == 0) && (min_res == 0)) {
            return `${h}h`;
        } else if ((max_res == 0) && (min_res == 1)) {
            return `${h}h${pm}`;
        } else if ((max_res == 0) && (min_res == 2)) {
            return `${h}h${pm}m${ps}`;
        } else if ((max_res == 1) && (min_res == 1)) {
            return `${m}m`;
        } else if ((max_res == 1) && (min_res == 2)) {
            return `${m}m${ps}`;
        } else if ((max_res == 2) && (min_res == 2)) {
            return `${s}s`;
        }
    """)

def suffix_yaxis(f):
    f.yaxis.formatter = bom.FuncTickFormatter(code="""
        var min_diff = Infinity;
        for (var i = 0; i < ticks.length-1; i++) {
            min_diff = Math.min(min_diff, ticks[i+1]-ticks[i]);
        }

        var suffixes = [
            'y', 'z', 'a', 'f', 'p', 'n', 'µ', 'm',
            '', 
            'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'];
        var precision = Math.floor(Math.log10(min_diff));
        var scale = Math.floor(precision/3);
        var index = scale + 8;
        if (index < 0) {
            //TODO: Fall back to numbro here
            return tick;
        } else if (index == 7) {
            // Millis are weird. Feels better to rende them as decimals.
            var decimals = -precision;
            return `${tick.toFixed(decimals)}`
        } else if (index < suffixes.length) {
            var suffix = suffixes[index];
            var scaled = tick/Math.pow(10, 3*scale);
            return `${scaled.toFixed(0)}${suffix}`
        } else {
            //TODO: Fall back to numbro here
            return tick;
        }
    """)

def x_zeroline(f):
    f.add_layout(bom.Span(location=0, dimension='height'))

def default_tools(f):
    f.toolbar_location = None
    f.toolbar.active_drag = f.select_one(bom.BoxZoomTool)
    # f.toolbar.active_scroll = f.select_one(bom.WheelZoomTool)
    # f.toolbar.active_inspect = f.select_one(bom.HoverTool)
    f.js_on_event(
        boe.DoubleTap, 
        bom.callbacks.CustomJS(args=dict(p=f), code='p.reset.emit()'))

def styling(f):
    timedelta_xaxis(f)
    suffix_yaxis(f)

def legend(f):
    f.legend.label_text_font_size = '8pt'
    f.legend.margin = 7
    f.legend.padding = 0
    f.legend.spacing = 0
    f.legend.background_fill_alpha = 0.3
    f.legend.border_line_alpha = 0.
    f.legend.location = 'top_left'

def align(readers, rule):
    df = {}
    for k, r in readers.items():
        df[k] = r.resample(**dict(r.pandas()), rule=rule)
    df = pd.concat(df, 1)
    df.index = df.index - df.index[0]
    return df.reset_index()

class Simple:
    fig_kwargs = {}
    line_kwargs = {'width': 2}

    def __init__(self, readers, rule):
        self.readers = readers
        self.rule = rule

        aligned = align(self.readers, self.rule)
        self.source = bom.ColumnDataSource(aligned)

        f = bop.figure(
            x_range=bom.DataRange1d(start=0, follow='end'), 
            tooltips=[('', '$data_y')], 
            **self.fig_kwargs)

        for key, color in zip(readers, cycle(Category10_10)):
            s = split(key)
            label = dict(legend_label=s.label) if s.label else dict()
            f.line(
                x='_time', 
                y=key, 
                color=color, 
                source=self.source, 
                **label,
                **self.line_kwargs)

        default_tools(f)
        x_zeroline(f)
        styling(f)

        s = split(list(readers)[0])
        if s.label:
            legend(f)
        f.title = bom.Title(text=s.title)

        self.figure = f

    def refresh(self):
        aligned = align(self.readers, self.rule)
        threshold = len(self.source.data['_time'])
        new = aligned.iloc[threshold:]
        self.source.stream(new)

class Log(Simple):
    fig_kwargs = {'y_axis_type': 'log'}

    def __init__(self, readers, rule):
        super().__init__(readers, rule)
        self.figure.formatter = bom.LogTickFormatter()

class Confidence:

    def __init__(self, readers, rule):
        self.readers = readers
        self.rule = rule

        aligned = self.aligned()
        self.source = bom.ColumnDataSource(aligned)

        f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])

        for key, color in zip(readers, cycle(Category10_10)):
            s = split(key)
            label = dict(legend_label=s.label) if s.label else dict()
            f.varea(
                x='_time', y1=f'{key}.μ-', y2=f'{key}.μ+', 
                color=color, alpha=.2, source=self.source, **label)
            f.line(
                x='_time', y=f'{key}.μ', 
                color=color, source=self.source, **label)

        default_tools(f)
        x_zeroline(f)
        styling(f)
        s = split(list(readers)[0])
        if s.label:
            legend(f)
        f.title = bom.Title(text=s.title)

        self.figure = f

    def aligned(self):
        aligned = align(self.readers, self.rule)
        aligned.columns = ['.'.join(d for d in c if d) for c in aligned.columns]
        return aligned

    def refresh(self):
        aligned = self.aligned()
        threshold = len(self.source.data['_time'])
        new = aligned.iloc[threshold:]
        self.source.stream(new)
