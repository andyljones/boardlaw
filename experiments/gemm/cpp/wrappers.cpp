#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "../../../boardlaw/cpp/common.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

TT linear(TT& W, const TT& x, const TT& b, const TT& idxs) {
    // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/cuda/LinearAlgebra.cu#L155
    TORCH_CHECK(W.dim() == 3, "W must be a 3D tensor");
    TORCH_CHECK(x.dim() == 2, "x must be a 2D tensor");
    TORCH_CHECK(b.dim() == 2, "b must be a 2D tensor");
    TORCH_CHECK(idxs.dim() == 1, "idxs must be a 1D tensor");

    TORCH_CHECK(W.size(0) == x.size(0), "W dim 0 must match x dim 0");
    TORCH_CHECK(W.size(0) == b.size(0), "W dim 0 must match b dim 0");
    TORCH_CHECK(W.size(2) == x.size(1), "W dim 2 must match x dim 1");
    TORCH_CHECK(W.size(1) == b.size(1), "W dim 1 must match b dim 1");

    //TODO: Write a macro for these
    TORCH_CHECK_TYPE(W.scalar_type() == at::kFloat, "W expected ", toString(at::kFloat), " got ", toString(W.scalar_type()));
    TORCH_CHECK_TYPE(x.scalar_type() == at::kFloat, "x expected ", toString(at::kFloat), " got ", toString(x.scalar_type()));
    TORCH_CHECK_TYPE(b.scalar_type() == at::kFloat, "b expected ", toString(at::kFloat), " got ", toString(b.scalar_type()));
    TORCH_CHECK_TYPE(idxs.scalar_type() == at::kInt, "idxs expected ", toString(at::kInt), " got ", toString(idxs.scalar_type()));

    auto y = at::empty_like(x);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("linear", &linear, "W"_a, "x"_a, "b"_a, "idxs"_a, py::call_guard<py::gil_scoped_release>());
}