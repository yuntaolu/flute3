#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "flute.h"

namespace py = pybind11;
using namespace Flute;

PYBIND11_MODULE(pyflute, m) {
    m.doc() = "Python bindings for FLUTE - Fast Lookup-Table Based Rectilinear Steiner Minimal Tree Algorithm";

    // Bind Branch struct
    py::class_<Branch>(m, "Branch")
        .def(py::init<>())
        .def_readwrite("x", &Branch::x)
        .def_readwrite("y", &Branch::y)
        .def_readwrite("n", &Branch::n);

    // Bind Tree struct
    py::class_<Tree>(m, "Tree")
        .def(py::init<>())
        .def_readwrite("deg", &Tree::deg)
        .def_readwrite("length", &Tree::length)
        .def_property_readonly("branches", [](const Tree& t) {
            return py::array_t<Branch>(t.deg, t.branch);
        });

    // Bind functions
    m.def("readLUT", &readLUT, "Initialize FLUTE lookup tables");
    m.def("deleteLUT", &deleteLUT, "Clean up FLUTE lookup tables");
    
    m.def("flute_wl", [](py::array_t<DTYPE> x, py::array_t<DTYPE> y, int acc) {
        if (x.size() != y.size()) {
            throw std::runtime_error("x and y arrays must have the same length");
        }

        // Create mutable copies of the input arrays
        std::vector<DTYPE> x_copy(x.data(), x.data() + x.size());
        std::vector<DTYPE> y_copy(y.data(), y.data() + y.size());
        
        return flute_wl(x.size(), x_copy.data(), y_copy.data(), acc);
    }, "Calculate wirelength of RSMT", 
    py::arg("x"), py::arg("y"), py::arg("accuracy") = FLUTE_ACCURACY);

    m.def("flute", [](py::array_t<DTYPE> x, py::array_t<DTYPE> y, int acc) {
        if (x.size() != y.size()) {
            throw std::runtime_error("x and y arrays must have the same length");
        }

        // Create mutable copies of the input arrays
        std::vector<DTYPE> x_copy(x.data(), x.data() + x.size());
        std::vector<DTYPE> y_copy(y.data(), y.data() + y.size());
        
        return flute(x.size(), x_copy.data(), y_copy.data(), acc);
    }, "Construct RSMT", 
    py::arg("x"), py::arg("y"), py::arg("accuracy") = FLUTE_ACCURACY);

    m.def("free_tree", &free_tree, "Free the memory allocated for a tree");

    // Add write_svg binding
    m.def("write_svg", &write_svg, "Write tree to SVG file",
        py::arg("tree"), py::arg("filename"));
}