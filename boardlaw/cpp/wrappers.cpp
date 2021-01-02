#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "common.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;
using namespace torch::indexing;

TT mul(TT X, TT W, std::vector<int> starts, std::vector<int> ends, int repeats) {
    for (int i=0; i<W.size(0); i++) {
        auto stream = at::cuda::getStreamFromPool(X.device().index());
        at::cuda::CUDAStreamGuard guard(stream);
        auto s = Slice(starts[i], ends[i], None);
        for (int r=0; r<repeats; r++) {
            X.index_put_({s}, at::matmul(X.index({s}), W[i]));
        }
    }
    return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("mul", &mul, "X"_a, "W"_a, "starts"_a, "ends"_a, "repeats"_a, py::call_guard<py::gil_scoped_release>());
}

