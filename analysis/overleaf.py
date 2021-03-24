import plotnine as pn
import matplotlib.pyplot as plt
from io import BytesIO
import json
from pathlib import Path

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

def plot(x, name):
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

def table(s, name):
    _dropbox_upload(s.encode(), f'/Apps/Overleaf/boardlaw/tables/{name}.tex')
