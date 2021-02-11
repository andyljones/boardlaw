import sys
import torch
import torch.cuda
import sysconfig
from pkg_resources import resource_filename

DEBUG = False

def load_cuda(pkg, files):
    # This import is pretty slow, so let's defer it
    import torch.utils.cpp_extension

    name = pkg.split('.')[-1] + 'cuda' 
    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')
    return torch.utils.cpp_extension.load(
        name=name, 
        sources=[resource_filename(pkg, f'cpp/{fn}') for fn in files], 
        extra_cflags=['-std=c++17'] + (['-g'] if DEBUG else []), 
        extra_cuda_cflags=['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else []),
        #TODO: Don't hardcode this
        extra_include_paths=['/usr/local/cuda/include'],
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

def load_cpu(pkg, files):
    # This import is pretty slow, so let's defer it
    import torch.utils.cpp_extension

    name = pkg.split('.')[-1] + 'cuda' 
    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')
    return torch.utils.cpp_extension.load(
        name=name, 
        sources=[resource_filename(pkg, f'cpp/{fn}') for fn in files], 
        extra_cflags=['-std=c++17', '-DNOCUDA'] + (['-g'] if DEBUG else []), 
        with_cuda=False,
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

_has_cuda = None
def load(pkg, files=('wrappers.cpp', 'kernels.cu')):

    # This bit of madness is because of this bug: https://github.com/pytorch/pytorch/issues/52145
    global _has_cuda
    if _has_cuda is None:
        try:
            torch.cuda.init()
        except RuntimeError:
            _has_cuda = False
        else:
            _has_cuda = True

    if _has_cuda:
        return load_cuda(pkg, files)
    else:
        return load_cpu(pkg, [f for f in files if not f.endswith('.cu')])


def assert_shape(x, s):
    assert (x.ndim == len(s)) and x.shape == s, f'Expected {s}, got {x.shape}'

