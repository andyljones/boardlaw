#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "common.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("step", &step, "board"_a, "seats"_a, "actions"_a, py::call_guard<py::gil_scoped_release>());
    m.def("observe", &observe, "board"_a, "seats"_a, py::call_guard<py::gil_scoped_release>());
    m.def("observe_old", &observe_old, "board"_a, "seats"_a, py::call_guard<py::gil_scoped_release>());
}
