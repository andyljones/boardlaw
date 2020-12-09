import pandas as pd

def format(v):
    if isinstance(v, int):
        return f'{v}'
    if isinstance(v, float):
        return f'{v:.6g}'
    if isinstance(v, list):
        return ', '.join(format(vv) for vv in v)
    if isinstance(v, dict):
        return '{' + ', '.join(f'{k}: {format(vv)}' for k, vv in v.items()) + '}'
    return str(v)

def single(finals, info):
    is_multi = len(info) > 1
    (title,) = info.title.unique()
    ks, vs = ([f'{title}/'], ['']) if is_multi else ([], [])
    for (category, name), row in info.iterrows():
        ks.append(f'  {row.label}' if is_multi else row.title)
        vs.append(format(finals[category, name]))
    return ks, vs

def confidence(finals, info):
    is_multi = len(info) > 1
    (title,) = info.title.unique()
    info = pd.concat([info, info.label.str.extract('^(?P<seq>.*)/(?P<stat>.*)')], 1)
    ks, vs = ([f'{title}/'], ['']) if is_multi else ([], [])
    for (category, title, seq), i in info.groupby(['category', 'title', 'seq']):
        i = i.set_index('stat')
        μ, σ = finals[category, i.loc['μ', 'key']], finals[category, i.loc['σ', 'key']]
        ks.append(f'  {title}/{seq}' if is_multi else f'{title}')
        vs.append(f'{μ:.2f}±{2*σ:.2f}')
    return ks, vs