from collections import namedtuple

from torch import tensor
from . import dotdict
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

def children(v):
    if hasattr(v, 'next_functions'):
        for (u, _) in v.next_functions:
            if u is not None:
                yield u
    if hasattr(v, 'saved_tensors'):
        for u in v.saved_tensors:
            yield u

def traverse(output):

    seen = set()
    boundary = {v.grad_fn for v in output} if isinstance(output, tuple) else {output.grad_fn}
    while boundary:
        v = boundary.pop()
        if v not in seen:
            seen.add(v)
            for c in children(v):
                boundary.add(c)

            yield v

def grads(l):

    hooks = {}
    tensorgrads, ingrads, outgrads = {}, {}, {}
    def add_hook(v): 
        
        def tensorhook(g):
            tensorgrads[str(id(v))] = g.clone()
            
        def funchook(i, o):
            ingrads[str(id(v))] = i.clone()
            outgrads[str(id(v))] = o.clone()
            
        hook = tensorhook if isinstance(v, torch.Tensor) else funchook
        hooks[str(id(v))] = v.register_hook(hook)

    for v in traverse(l):
        add_hook(v)
    l.backward()

    for _, h in hooks:
        h.remove()

    return dotdict.dotdict(tensors=tensorgrads, ins=ingrads, outs=outgrads)

def size_to_str(size):
    return '(' + (', ').join(['%d' % v for v in size]) + ')'

def resize_graph(dot, size_per_element=0.15, min_size=12):
    """Resize the graph according to how much content it contains.
    Modify the graph in place.
    """
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

def graph(output, params={}):
    if not isinstance(output, tuple):
        output = (output,)
    if isinstance(params, list):
        params = dict(params)
    assert all(isinstance(p, Variable) for p in params.values())

    param_map = {id(v): k for k, v in params.items()}
    output_nodes = {v.grad_fn for v in output}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

    for v in traverse(output):
        if torch.is_tensor(v):
            # note: this used to show .saved_tensors in pytorch0.2, but stopped
            # working as it was moved to ATen and Variable-Tensor merged
            dot.node(str(id(v)), size_to_str(v.size()), fillcolor='orange')
        elif hasattr(v, 'variable'):
            u = v.variable
            name = param_map.get(id(u), '') 
            node_name = '%s\n %s' % (name, size_to_str(u.size()))
            dot.node(str(id(v)), node_name, fillcolor='lightblue')
        elif v in output_nodes:
            dot.node(str(id(v)), str(type(v).__name__), fillcolor='darkolivegreen1')
        else:
            dot.node(str(id(v)), str(type(v).__name__))

        for c in children(v):
            dot.edge(str(id(c)), str(id(v)))

    resize_graph(dot)

    return dot

def demo():
    from scalinglaws.arena.vb import VB
    vb = VB(5)

    N = 5
    w = torch.zeros((N, N)).int()
    n = torch.zeros((N, N)).int()
    l = vb(n, w)
    grads(l)

    graph(l, vb.named_parameters())