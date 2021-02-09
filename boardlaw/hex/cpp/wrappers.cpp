#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "common.h"
#include "cpu.cpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

#ifdef NOCUDA
TT step(TT board, TT seats, TT actions) {
    return hexcpu::step(board, seats, actions);
}

TT observe(TT board, TT seats) {
    return hexcpu::observe(board, seats);
}
#else
TT step(TT board, TT seats, TT actions) {
    if (board.device().is_cuda()) {
        return hexcuda::step(board, seats, actions);
    } else {
        return hexcpu::step(board, seats, actions);
    }
}

TT observe(TT board, TT seats) {
    if (board.device().is_cuda()) {
        return hexcuda::observe(board, seats);
    } else {
        return hexcpu::observe(board, seats);
    }
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("step", &step, "board"_a, "seats"_a, "actions"_a, py::call_guard<py::gil_scoped_release>());
    m.def("observe", &observe, "board"_a, "seats"_a, py::call_guard<py::gil_scoped_release>());
}
