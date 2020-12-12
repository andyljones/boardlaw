import torch
import sysconfig
from pkg_resources import resource_filename
import torch.utils.cpp_extension

DEBUG = False

def _cuda():
    [torch_libdir] = torch.utils.cpp_extension.library_paths()
    python_libdir = sysconfig.get_config_var('LIBDIR')
    libpython_ver = sysconfig.get_config_var('LDVERSION')
    return torch.utils.cpp_extension.load(
        name='scalinglawscuda', 
        sources=[resource_filename(__package__, f'cpp/{fn}') for fn in ('wrappers.cpp', 'kernels.cu')], 
        extra_cflags=['-std=c++17'] + (['-g'] if DEBUG else []), 
        extra_cuda_cflags=['--use_fast_math', '-lineinfo', '-std=c++14'] + (['-g', '-G'] if DEBUG else []),
        extra_ldflags=[
            f'-lpython{libpython_ver}', '-ltorch', '-ltorch_python', '-lc10_cuda', '-lc10', 
            f'-L{torch_libdir}', f'-Wl,-rpath,{torch_libdir}',
            f'-L{python_libdir}', f'-Wl,-rpath,{python_libdir}'])

cuda = _cuda()
for k in dir(cuda):
    if not k.startswith('__'):
        globals()[k] = getattr(cuda, k)

def assert_shape(x, s):
    assert (x.ndim == len(s)) and x.shape == s, f'Expected {s}, got {x.shape}'
    assert x.device.type == 'cuda', f'Expected CUDA tensor, got {x.device.type}'

def mcts(logits, w, n, c_puct, seats, terminal, children):
    B, T, A = logits.shape
    S = w.shape[-1]
    assert_shape(w, (B, T, S))
    assert_shape(n, (B, T))
    assert_shape(c_puct, (B,))
    assert_shape(seats, (B, T))
    assert_shape(terminal, (B, T))
    assert_shape(children, (B, T, A))
    assert (c_puct > 0.).all(), 'Zero c_puct not supported; will lead to an infinite loop in the kernel'

    with torch.cuda.device(logits.device):
        return cuda.MCTS(logits, w, n.int(), c_puct, seats.int(), terminal, children.int())

