#include <ATen/ATen.h>
#include <variant>
#include <exception>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace pybind11::literals;

using TT = at::Tensor;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define our own copy of RestrictPtrTraits here, as the at::RestrictPtrTraits is 
// only included during NVCC compilation, not plain C++. This would mess things up 
// since this file is included on both the NVCC and Clang sides. 
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

template<typename T>
at::ScalarType dtype() { return at::typeMetaToScalarType(caffe2::TypeMeta::Make<T>()); }

template <typename T, size_t D>
struct TensorProxy {

    using PTA = at::PackedTensorAccessor32<T, D, RestrictPtrTraits>;
    TT t; 

    TensorProxy(const at::Tensor t) : t(t) {
        CHECK_INPUT(t);
        TORCH_CHECK_TYPE(t.scalar_type() == dtype<T>(), "expected ", toString(dtype<T>()), " got ", toString(t.scalar_type()));
        TORCH_CHECK(t.ndimension() == D, "expected ", typeid(D).name(), " got ", "t.ndimension()");
    }

    PTA pta() const { return t.packed_accessor32<T, D, RestrictPtrTraits>(); }

    size_t size(const size_t i) const { return t.size(i); }
};

using F1D = TensorProxy<float, 1>;
using F2D = TensorProxy<float, 2>;
using F3D = TensorProxy<float, 3>;
using F4D = TensorProxy<float, 4>;

using H1D = TensorProxy<at::Half, 1>;
using H2D = TensorProxy<at::Half, 2>;
using H3D = TensorProxy<at::Half, 3>;
using H4D = TensorProxy<at::Half, 4>;

using I1D = TensorProxy<int, 1>;
using I2D = TensorProxy<int, 2>;
using I3D = TensorProxy<int, 3>;

using S1D = TensorProxy<short, 1>;
using S2D = TensorProxy<short, 2>;
using S3D = TensorProxy<short, 3>;

using B1D = TensorProxy<bool, 1>;
using B2D = TensorProxy<bool, 2>;
using B3D = TensorProxy<bool, 3>;

using C1D = TensorProxy<uint8_t, 1>;
using C2D = TensorProxy<uint8_t, 2>;
using C3D = TensorProxy<uint8_t, 3>;
using C4D = TensorProxy<uint8_t, 4>;