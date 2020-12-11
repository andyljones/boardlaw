#include <ATen/ATen.h>
#include <variant>
#include <exception>
#include <pybind11/pybind11.h>

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

struct DescentResult {
  const TT parents;
  const TT actions;
};

TT solve_policy(const TT pi, const TT q, const TT lambda_n);
DescentResult descend(
    const TT logits, const TT w, const TT n, const TT c_puct,
    const TT seats, const TT terminal, const TT children);