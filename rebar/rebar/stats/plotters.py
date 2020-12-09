import pandas as pd
from bokeh import models as bom
from bokeh import plotting as bop
from bokeh import io as boi
from bokeh import layouts as bol
from bokeh import events as boe

from bokeh.palettes import Category10_10
from itertools import cycle

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

def timeseries(source, info):
    y = info.id.iloc[0]
    #TODO: Work out how to apply the axes formatters to the tooltips
    f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])
    f.line(x='time_', y=y, source=source)
    default_tools(f)
    x_zeroline(f)
    styling(f)

    return f

def timedataframe(source, info):
    f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])

    for y, label, color in zip(info.id.tolist(), info.label.tolist(), cycle(Category10_10)):
        f.line(x='time_', y=y, legend_label=label, color=color, width=2, source=source)

    default_tools(f)
    x_zeroline(f)
    styling(f)

    f.legend.label_text_font_size = '8pt'
    f.legend.margin = 7
    f.legend.padding = 0
    f.legend.spacing = 0
    f.legend.background_fill_alpha = 0.3
    f.legend.border_line_alpha = 0.
    f.legend.location = 'top_left'

    return f

def single(source, info):
    # import aljpy; aljpy.extract()
    if (len(info) == 1) and (info.label == '').all():
        return timeseries(source, info)
    else:
        return timedataframe(source, info)


def confidence(source, info):
    f = bop.figure(x_range=bom.DataRange1d(start=0, follow='end'), tooltips=[('', '$data_y')])

    info = pd.concat([info, info.label.str.extract('^(?P<seq>.*)/(?P<stat>.*)')], 1)

    for (seq, i), color in zip(info.groupby('seq'), cycle(Category10_10)):
        i = i.set_index('stat')['id']
        f.varea(x='time_', y1=i['μ-'], y2=i['μ+'], legend_label=seq, color=color, alpha=.5, source=source)
        f.line(x='time_', y=i['μ'], legend_label=seq, color=color, source=source)

    default_tools(f)
    x_zeroline(f)
    styling(f)

    f.legend.label_text_font_size = '8pt'
    f.legend.margin = 7
    f.legend.padding = 0
    f.legend.spacing = 0
    f.legend.background_fill_alpha = 0.3
    f.legend.border_line_alpha = 0.
    f.legend.location = 'top_left'

    return f