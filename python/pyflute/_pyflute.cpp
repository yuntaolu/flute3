// python/pyflute.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../../flute.h"  // header path

namespace py = pybind11;
using namespace Flute;

PYBIND11_MODULE(_pyflute, m) {  // Changed name to _pyflute for better Python packaging
    m.doc() = "Python bindings for FLUTE - Fast Look-Up Table Based Rectilinear Steiner Minimal Tree Algorithm";

    // Branch class
    py::class_<Branch>(m, "Branch")
        .def(py::init<>())
        .def_readwrite("x", &Branch::x)
        .def_readwrite("y", &Branch::y)
        .def_readwrite("n", &Branch::n)
        .def("__repr__",
            [](const Branch& b) {
                return "Branch(x=" + std::to_string(b.x) + 
                       ", y=" + std::to_string(b.y) + 
                       ", n=" + std::to_string(b.n) + ")";
            });

    // Tree class
    py::class_<Tree>(m, "Tree")
        .def(py::init<>())
        .def_readwrite("deg", &Tree::deg)
        .def_readwrite("length", &Tree::length)
        .def_property_readonly("branches", [](const Tree& t) {
            return py::array_t<Branch>(t.deg, t.branch);
        })
        .def("__repr__",
            [](const Tree& t) {
                return "Tree(degree=" + std::to_string(t.deg) + 
                       ", length=" + std::to_string(t.length) + ")";
            });

    // Module functions
    m.def("readLUT", &readLUT, "Initialize FLUTE lookup tables");
    m.def("deleteLUT", &deleteLUT, "Clean up FLUTE lookup tables");
    
    m.def("flute_wl", [](py::array_t<DTYPE> x, py::array_t<DTYPE> y, int acc) {
        if (x.size() != y.size()) {
            throw std::runtime_error("x and y arrays must have the same length");
        }
        std::vector<DTYPE> x_copy(x.data(), x.data() + x.size());
        std::vector<DTYPE> y_copy(y.data(), y.data() + y.size());
        return flute_wl(x.size(), x_copy.data(), y_copy.data(), acc);
    }, "Calculate wirelength of RSMT", 
    py::arg("x"), py::arg("y"), py::arg("accuracy") = FLUTE_ACCURACY);

    m.def("flute", [](py::array_t<DTYPE> x, py::array_t<DTYPE> y, int acc) {
        if (x.size() != y.size()) {
            throw std::runtime_error("x and y arrays must have the same length");
        }
        std::vector<DTYPE> x_copy(x.data(), x.data() + x.size());
        std::vector<DTYPE> y_copy(y.data(), y.data() + y.size());
        return flute(x.size(), x_copy.data(), y_copy.data(), acc);
    }, "Construct RSMT", 
    py::arg("x"), py::arg("y"), py::arg("accuracy") = FLUTE_ACCURACY);

    // Bind the flute_pin_wl function
    // m.def("flute_pin_wl", &Flute::flute_pin_wl,
    //       py::arg("tree"),
    //       py::arg("x1"),
    //       py::arg("y1"),
    //       py::arg("x2"),
    //       py::arg("y2"),
    //       "Calculate the path length between two points on the Steiner tree.\n\n"
    //       "Args:\n"
    //       "  - tree (Tree): The Steiner tree\n"
    //       "  - x1 (int): x-coordinate of first point\n"
    //       "  - y1 (int): y-coordinate of first point\n"
    //       "  - x2 (int): x-coordinate of second point\n"
    //       "  - y2 (int): y-coordinate of second point\n\n"
    //       "Returns:\n"
    //       "  int: Total path length between the points following the Steiner tree structure.\n"
    //        "  Returns -1 if no valid path is found or if input is invalid.\n");
    
    // Bind the write_svg function
    m.def("write_svg", &write_svg, 
          py::arg("tree"),
          py::arg("filename"),
          "Write a Steiner tree to an SVG file.\n\n"
          "Args:\n"
          "    tree (Tree): The Steiner tree to visualize\n"
          "    filename (str): Output SVG file path\n\n"
          "The SVG will show:\n"
          "- Red circles for pins\n"
          "- Blue circles for Steiner points\n"
          "- Black lines for connections");
}