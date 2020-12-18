from cloudpickle.cloudpickle_fast import (
    CloudPickler, subimport, dynamic_subimport,
    _BUILTIN_TYPE_NAMES, _builtin_type, _dynamic_class_reduce,
    types)
from cloudpickle.cloudpickle import _lookup_module_and_qualname
from collections import ChainMap
import sys

def _is_importable(obj, name=None):
    """Dispatcher utility to test the importability of various constructs."""
    if isinstance(obj, types.FunctionType):
        return _lookup_module_and_qualname(obj, name=name) is not None
    elif issubclass(type(obj), type):
        return _lookup_module_and_qualname(obj, name=name) is not None
    elif isinstance(obj, types.ModuleType):
        # We assume that sys.modules is primarily used as a cache mechanism for
        # the Python import machinery. Checking if a module has been added in
        # is sys.modules therefore a cheap and simple heuristic to tell us whether
        # we can assume  that a given module could be imported by name in
        # another Python process.
        return obj.__name__ in sys.modules
    else:
        raise TypeError(
            "cannot check importability of {} instances".format(
                type(obj).__name__)
        )

def _module_reduce(obj):
    if _is_importable(obj):
        return subimport, (obj.__name__,)
    else:
        obj.__dict__.pop('__builtins__', None)
        return dynamic_subimport, (obj.__name__, vars(obj))

def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    if obj is type(None):  # noqa
        return type, (None,)
    elif obj is type(Ellipsis):
        return type, (Ellipsis,)
    elif obj is type(NotImplemented):
        return type, (NotImplemented,)
    elif obj in _BUILTIN_TYPE_NAMES:
        return _builtin_type, (_BUILTIN_TYPE_NAMES[obj],)
    elif not _is_importable(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented

class LocalPickler(CloudPickler):

    _dispatch_table = ChainMap({types.ModuleType: _module_reduce}, CloudPickler._dispatch_table)

    def _function_reduce(self, obj):
        if _is_importable(obj):
            return NotImplemented
        else:
            return self._dynamic_function_reduce(obj)

    def reducer_override(self, obj):
        t = type(obj)
        try:
            is_anyclass = issubclass(t, type)
        except TypeError:  # t is not a class (old Boost; see SF #502085)
            is_anyclass = False

        if is_anyclass:
            return _class_reduce(obj)
        elif isinstance(obj, types.FunctionType):
            return self._function_reduce(obj)
        else:
            # fallback to save_global, including the Pickler's
            # distpatch_table
            return NotImplemented

OLD = """
from torch import nn
class TestNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.w(x)*0 + 1
"""
NEW = """
from torch import nn
class TestNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Linear(2, 2)
    
    def forward(self, x):
        return self.w(x)*0 + 2
"""

def test():
    import os
    import pickle
    import pathlib
    import importlib
    import torch

    try:
        path = pathlib.Path('test_pickling.py')
        import test_pickling

        os.remove('__pycache__/test_pickling.cpython-38.pyc')
        path.write_text(OLD)
        importlib.reload(test_pickling)
        from test_pickling import TestNetwork


        net = TestNetwork()
        with open('test_pickling.pkl', 'wb+') as f:
            LocalPickler(f).dump(net)

        os.remove('__pycache__/test_pickling.cpython-38.pyc')
        path.write_text(NEW)
        importlib.reload(test_pickling)
        from test_pickling import TestNetwork

        with open('test_pickling.pkl', 'rb+') as f:
            old = pickle.load(f)
            
        new = TestNetwork()
            
        assert old.w.weight.shape[0] == 1
        assert new.w.weight.shape[0] == 2

        assert old(torch.zeros(1)) == 1
        assert new(torch.zeros(2)) == 2
    finally:
        os.remove('test_pickling.pkl')
        os.remove('test_pickling.py')