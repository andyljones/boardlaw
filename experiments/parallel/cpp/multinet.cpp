#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "../../../boardlaw/cpp/common.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <future>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;
using namespace torch::indexing;

using Weights = std::vector<std::vector<TT>>;
using Slices = std::vector<std::tuple<int, int>>;

void worker(TT X, Weights Ws, Weights bs, Slices slices, int m, std::promise<TT> promise) {
    torch::NoGradGuard gradguard;

    auto stream = at::cuda::getStreamFromPool(X.device().index());
    at::cuda::CUDAStreamGuard streamguard(stream);
    auto s = Slice(std::get<0>(slices[m]), std::get<1>(slices[m]), None);

    auto Xs = X.index({s});
    for (int l=0; l<Ws[m].size(); l++) {
        Xs = at::addmm(bs[m][l], Xs, Ws[m][l]);
        Xs = torch::relu(Xs);
    }
    promise.set_value(Xs);
}

TT forward(TT X, Weights Ws, Weights bs, Slices slices) {
    std::vector<TT> parts = {};
    std::vector<std::future<TT>> futures = {};
    std::vector<std::thread> threads = {};
    for (int m=0; m<Ws.size(); m++) {
        std::promise<TT> promise;
        futures.push_back(promise.get_future());
        threads.push_back(std::thread(worker, X, Ws, bs, slices, m, std::move(promise)));
    }

    for (int m=0; m<Ws.size(); m++) {
        parts.push_back(futures[m].get());
        threads[m].join();
    }
    at::cuda::device_synchronize();
    return at::cat(parts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "X"_a, "Ws"_a, "bs"_a, "slices"_a, py::call_guard<py::gil_scoped_release>());
}

