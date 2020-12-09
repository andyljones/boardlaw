#include <ATen/ATen.h>
#include <variant>
#include <exception>
#include <iostream>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace pybind11::literals;

using TT = at::Tensor;

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
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
        AT_ASSERT(t.scalar_type() == dtype<T>());
        AT_ASSERT(t.ndimension() == D);
    }

    static TensorProxy<T, D> empty(at::IntArrayRef size) { return TensorProxy(at::empty(size, at::device(at::kCUDA).dtype(dtype<T>()))); }
    static TensorProxy<T, D> zeros(at::IntArrayRef size) { return TensorProxy(at::zeros(size, at::device(at::kCUDA).dtype(dtype<T>()))); }
    static TensorProxy<T, D> ones(at::IntArrayRef size) { return TensorProxy(at::ones(size, at::device(at::kCUDA).dtype(dtype<T>()))); }

    PTA pta() const { return t.packed_accessor32<T, D, RestrictPtrTraits>(); }

    size_t size(const size_t i) const { return t.size(i); }
};

using TP1D = TensorProxy<float, 1>;
using TP2D = TensorProxy<float, 2>;

struct Solution {
    const TP2D policy;
    const TP1D alpha_min;
    const TP1D alpha_star;
    const TP1D error;

    Solution(const uint B, const uint A) : 
        policy(TP2D::empty({B, A})),
        alpha_min(TP1D::empty({B})),
        alpha_star(TP1D::empty({B})),
        error(TP1D::empty({B})) {
    } 
};

Solution solve_policy(const TT pi, const TT q, const TT lambda_n);