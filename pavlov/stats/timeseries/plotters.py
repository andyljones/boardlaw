import numpy as np
import pandas as pd
from bokeh import models as bom
from bokeh import plotting as bop
from bokeh import io as boi
from bokeh import layouts as bol
from bokeh import events as boe

from bokeh.palettes import Category10_10, Viridis256
from itertools import cycle

from pandas.core.series import Series
from .. import registry

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
    for reader in readers:
        df[reader.prefix] = reader.resample(rule=rule)
    df = pd.concat(df, 1)
    df.index.name = '_time'
    # Drop the last row since it represents an under-full window.
    return df.reset_index().iloc[:-1]

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

        for reader, color in zip(readers, cycle(Category10_10)):
            p = registry.parse_prefix(reader.prefix)
            label = dict(legend_label=p.label) if p.label else dict()
            f.line(
                x='_time', 
                y=reader.prefix, 
                color=color, 
                source=self.source, 
                **label,
                **self.line_kwargs)

        default_tools(f)
        x_zeroline(f)
        styling(f)

        p = registry.parse_prefix(readers[0].prefix)
        if p.label:
            legend(f)
        f.title = bom.Title(text=p.group)

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
        self.figure.yaxis[0].formatter = bom.LogTickFormatter()

class Percent(Simple):

    def __init__(self, readers, rule):
        super().__init__(readers, rule)
        self.figure.yaxis[0].formatter = bom.NumeralTickFormatter(format="0%")

class Confidence:

    def __init__(self, readers, rule):
        self.readers = readers
        self.rule = rule

        self.source = bom.ColumnDataSource(self.aligned())

        f = bop.figure(
            x_range=bom.DataRange1d(start=0, follow='end'), 
            tooltips=[('', '$data_y')])

        for reader, color in zip(readers, cycle(Category10_10)):
            p = registry.parse_prefix(reader.prefix)
            label = dict(legend_label=p.label) if p.label else dict()
            f.varea(
                x='_time', y1=f'{reader.prefix}.μ-', y2=f'{reader.prefix}.μ+', 
                color=color, alpha=.2, source=self.source, **label)
            f.line(
                x='_time', y=f'{reader.prefix}.μ', 
                color=color, source=self.source, **label)

        default_tools(f)
        x_zeroline(f)
        styling(f)
        p = registry.parse_prefix(readers[0].prefix)
        if p.label:
            legend(f)
        f.title = bom.Title(text=p.group)

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

class Quantiles:

    def __init__(self, readers, rule):
        [self.reader] = readers
        self.rule = rule

        aligned = self.aligned()
        self.source = bom.ColumnDataSource(aligned)

        f = bop.figure(
            x_range=bom.DataRange1d(start=0, follow='end', range_padding=0), 
            y_range=bom.DataRange1d(start=0),
            tooltips=[('', '$data_y')])

        p = registry.parse_prefix(self.reader.prefix)
        n_bands = aligned.shape[1] - 1
        assert n_bands % 2 == 1
        for i in range(n_bands):
            color = Viridis256[255 - 256*i//n_bands]
            lower = aligned.columns[i+1]
            f.line(x='_time', y=f'{lower}', color=color, source=self.source)
        

        default_tools(f)
        styling(f)
        p = registry.parse_prefix(readers[0].prefix)
        f.title = bom.Title(text=p.group)

        self.figure = f

    def aligned(self):
        df = self.reader.resample(rule=self.rule)
        df.index.name = '_time'
        # Drop the last row since it represents an under-full window.
        df = df.reset_index().iloc[:-1]
        return df

    def refresh(self):
        aligned = self.aligned()
        threshold = len(self.source.data['_time'])
        new = aligned.iloc[threshold:]
        self.source.stream(new)

class Null:

    def __init__(self, readers, **kwargs):
        f = bop.figure(
            tooltips=[('', '$data_y')])
        p = registry.parse_prefix(readers[0].prefix)
        f.title = bom.Title(text=p.group)
        default_tools(f)

        self.figure = f

    def refresh(self):
        pass

class Line:

    def __init__(self, readers, **kwargs):

        self.readers = readers
        self.source = bom.ColumnDataSource(self.combined())

        f = bop.figure(
            tooltips=[('', '$data_y')])

        for reader, color in zip(readers, cycle(Category10_10)):
            p = registry.parse_prefix(reader.prefix)
            label = dict(legend_label=p.label) if p.label else dict()
            f.line(
                x=f'{reader.prefix}.x', y=f'{reader.prefix}.y', 
                color=color, source=self.source, **label)
            f.circle(
                x=f'{reader.prefix}.x', y=f'{reader.prefix}.y', 
                color=color, source=self.source, **label)

        f.title = bom.Title(text=p.group)
        default_tools(f)

        self.figure = f

    def combined(self):
        combo = []
        for reader in self.readers:
            arrs = reader.array()
            combo.append(pd.DataFrame({
                f'{reader.prefix}.x': arrs['xs'], 
                f'{reader.prefix}.y': arrs['ys']}).sort_values(f'{reader.prefix}.x'))
        return pd.concat(combo, 1)

    def refresh(self):
        self.source.data = self.combined()