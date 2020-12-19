#include <torch/extension.h>
#include <torch/csrc/autograd/variable.h>
#include <pybind11/pybind11.h>
#include "common.h"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace std::string_literals;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::class_<Descent>(m, "Descent", py::module_local())
        .def_property_readonly("parents", [](Descent r) { return r.parents.t; })
        .def_property_readonly("actions", [](Descent r) { return r.actions.t; });
    
    py::class_<MCTS>(m, "MCTS", py::module_local())
        .def(py::init<TT, TT, TT, TT, TT, TT, TT>(), 
            "logits"_a, "w"_a, "n"_a, "c_puct"_a, "seats"_a, "terminal"_a, "children"_a)
        .def_property_readonly("logits", [](MCTS s)  { return s.logits.t; })
        .def_property_readonly("w", [](MCTS s)  { return s.w.t; })
        .def_property_readonly("n", [](MCTS s)  { return s.n.t; })
        .def_property_readonly("c_puct", [](MCTS s)  { return s.c_puct.t; })
        .def_property_readonly("seats", [](MCTS s)  { return s.seats.t; })
        .def_property_readonly("terminal", [](MCTS s)  { return s.terminal.t; })
        .def_property_readonly("children", [](MCTS s)  { return s.children.t; });
    
    py::class_<Backup>(m, "Backup", py::module_local())
        .def(py::init<TT, TT, TT, TT, TT, TT>(),
            "v"_a, "w"_a, "n"_a, "rewards"_a, "parents"_a, "terminal"_a);

    m.def("descend", &descend, "m"_a, py::call_guard<py::gil_scoped_release>());
    m.def("root", &root, "m"_a, py::call_guard<py::gil_scoped_release>());
    m.def("backup", &backup, "b"_a, "leaves"_a, py::call_guard<py::gil_scoped_release>());
}