import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotnine as pn
from io import BytesIO

def mpl_theme(width=12, height=8):
    return [
        pn.theme_matplotlib(),
        pn.theme(
            figure_size=(width, height), 
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1))
            ]

def poster_sizes():
    return pn.theme(
        text=pn.element_text(size=18),
        title=pn.element_text(size=18),
        legend_title=pn.element_text(size=18))

class IEEE(pn.theme):

    def __init__(self):
        # https://matplotlib.org/stable/tutorials/introductory/customizing.html
        margin = {'t': 1, 'b': 1, 'l': 1, 'r': 1, 'units': 'pt'}
        super().__init__(
            axis_title_x=pn.element_text(margin=margin), 
            axis_title_y=pn.element_text(margin=margin), 
            complete=True)

        self._rcParams.update({
            'figure.figsize': (3.487, 2.155),
            'figure.dpi': 300,
            'font.family': 'serif',
            'font.size': 6,
            'axes.grid': True,
            'axes.grid.which': 'both',
            'grid.linewidth': .25,
            'grid.alpha': .2,
            'axes.linewidth': .5,
            'axes.prop_cycle': mpl.rcParams['axes.prop_cycle']})
        

def ieee():
    # https://github.com/garrettj403/SciencePlots/blob/master/styles/journals/ieee.mplstyle
    return [
        pn.theme_mpl(),
        pn.theme(
            strip_background=pn.element_rect(color='w', fill='w'),
            panel_grid=pn.element_line(color='k', alpha=.1, size=.25),

            text=pn.element_text(family='serif', size=8),
            legend_title=pn.element_text(size=8),
            figure_size=(3.3, 2.5), 
            dpi=180,)]

def no_colorbar_ticks():
    return pn.guides(color=pn.guide_colorbar(ticks=False))

def _dropbox_upload(data, path):
    # https://www.dropbox.com/developers/documentation/http/documentation#files-upload
    import dropbox
    token = json.loads(Path('credentials.json').read_text())['dropbox']
    dbx = dropbox.Dropbox(token)

    # Will throw an UploadError if it fails
    dbx.files_upload(
        f=data, 
        path=path,
        mode=dropbox.files.WriteMode.overwrite)

def overleaf(x, name):
    if isinstance(x, pn.ggplot):
        fig = x.draw()
    elif isinstance(x, plt.Axes):
        fig = x.figure
    elif isinstance(x, plt.Figure):
        fig = x
    else:
        raise ValueError(f'Can\'t handle {type(x)}')
    
    bs = BytesIO()
    format = name.split('.')[-1]
    fig.savefig(bs, format=format, bbox_inches='tight', pad_inches=.005, dpi=600)
    plt.close(fig)

    _dropbox_upload(bs.getvalue(), f'/Apps/Overleaf/boardlaw/images/{name}')
