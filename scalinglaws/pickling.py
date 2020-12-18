"""This is a copy-pasted chunk of cloudpickle with a modified _is_importable that
redirects pickling of local-dir code from the standard pickling machinery (which
only records the name and file) into the cloudpickle machinery (which takes a full
copy of the code).
"""
from cloudpickle.cloudpickle_fast import (
    CloudPickler, subimport, dynamic_subimport,
    _BUILTIN_TYPE_NAMES, _builtin_type, _dynamic_class_reduce,
    types)
from cloudpickle.cloudpickle import _lookup_module_and_qualname
from collections import ChainMap
import sys
import os
from io import BytesIO

def is_library(obj, name):
    mod_qualname = _lookup_module_and_qualname(obj, name=name)
    if mod_qualname is None:
        return False
    else:
        mod, qualname = mod_qualname
        if hasattr(mod, '__file__'):
            return not mod.__file__.startswith(os.getcwd())
        return True

def _is_importable(obj, name=None):
    """This is the only function we've modified; everything else here is 
    just needed to nestle it into the CloudPickler's machinery"""
    if isinstance(obj, types.FunctionType):
        return is_library(obj, name)
    elif issubclass(type(obj), type):
        return is_library(obj, name)
    elif isinstance(obj, types.ModuleType):
        if hasattr(obj, '__file__'):
            if obj.__file__.startswith(os.getcwd()):
                return False
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

def dump(obj, file, protocol=None, buffer_callback=None):
    LocalPickler(file, protocol=protocol, buffer_callback=buffer_callback).dump(obj)

def dumps(obj, protocol=None, buffer_callback=None):
    with BytesIO() as file:
        cp = LocalPickler(file, protocol=protocol, buffer_callback=buffer_callback)
        cp.dump(obj)
        return file.getvalue()

### TESTS

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

def reload_recreate(path, contents):
    import importlib
    from pathlib import Path

    #TODO: Is there a better way to temporarily ignore the bytecode cache?
    Path('__pycache__/test_pickling.cpython-38.pyc').unlink(True)
    path.write_text(contents)
    import test_pickling
    importlib.reload(test_pickling)
    from test_pickling import TestNetwork
    return TestNetwork()

def test():
    import os
    import pickle
    import torch
    from pathlib import Path

    try:
        path = Path('test_pickling.py')

        old = reload_recreate(path, OLD)
        with open('test_pickling.pkl', 'wb+') as f:
            LocalPickler(f).dump(old)

        new = reload_recreate(path, NEW)
        with open('test_pickling.pkl', 'rb+') as f:
            old = pickle.load(f)
            
        assert old.w.weight.shape[0] == 1
        assert new.w.weight.shape[0] == 2

        assert old(torch.zeros(1))[0] == 1
        assert new(torch.zeros(2))[0] == 2
    finally:
        Path('test_pickling.pkl').unlink(True)
        Path('test_pickling.py').unlink(True)