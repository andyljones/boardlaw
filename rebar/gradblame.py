from collections import namedtuple
from . import dotdict
from graphviz import Digraph
import torch
from torch.autograd import Variable

Node = namedtuple('Node', ('name', 'inputs', 'attr', 'op'))

def traverse(output):

    seen = set()
    boundary = {v.grad_fn for v in output} if isinstance(output, tuple) else {output.grad_fn}
    while boundary:
        v = boundary.pop()
        if v not in seen:
            seen.add(v)
            if hasattr(v, 'next_functions'):
                for (u, _) in v.next_functions:
                    if u is not None:
                        boundary.add(u)
            if hasattr(v, 'saved_tensors'):
                for u in v.saved_tensors:
                    boundary.add(u)

            yield v

def grads(l):

    hooks = {}
    tensorgrads, ingrads, outgrads = {}, {}, {}
    def add_hook(v): 
        
        def tensorhook(g):
            tensorgrads[v] = g
            
        def funchook(i, o):
            ingrads[v] = i
            outgrads[v] = o
            
        if isinstance(v, torch.Tensor):
            hooks[v] = v.register_hook(tensorhook)
        else:
            hooks[v] = v.register_hook(funchook)

    for v in traverse(l):
        add_hook(v)
    l.backward()

    for _, h in hooks:
        h.remove()

    return dotdict.dotdict(tensors=tensorgrads, ins=ingrads, outs=outgrads)

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

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph.
    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # handle multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)

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
