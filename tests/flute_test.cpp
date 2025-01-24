// src/main.cpp
#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>
#include <flute.h>

using namespace Flute;

// Custom exception class for Flute errors
class FluteException : public std::runtime_error {
public:
    explicit FluteException(const std::string& msg) : std::runtime_error(msg) {}
};

// RAII wrapper for Flute LUT initialization/cleanup
class FluteLUTManager {
public:
    FluteLUTManager() {
        try {
            readLUT();  // Simply call readLUT() since it's void
        } catch (...) {
            throw FluteException("Failed to initialize Flute lookup tables");
        }
    }
    ~FluteLUTManager() {
        deleteLUT();
    }
    // Prevent copying
    FluteLUTManager(const FluteLUTManager&) = delete;
    FluteLUTManager& operator=(const FluteLUTManager&) = delete;
};

// Validate input arrays
void validateInputs(const std::vector<DTYPE>& x, const std::vector<DTYPE>& y) {
    if (x.empty() || y.empty()) {
        throw FluteException("Input arrays cannot be empty");
    }
    if (x.size() != y.size()) {
        throw FluteException("X and Y arrays must have the same size");
    }
    if (x.size() > 100) {  // Assuming max pins = 100
        throw FluteException("Too many pins (maximum 100 supported)");
    }
}

int main() {
    try {
        // RAII for LUT management
        FluteLUTManager lutManager;

        // Example: Calculate wirelength for 12 pins
        std::vector<DTYPE> x = {0, 100, 100, 0, 50, 80, 50, 0, 20, 10, 80, 60};
        std::vector<DTYPE> y = {0, 0, 100, 100, 90, 10, 80, 50, 10, 80, 50, 80};

        // Validate inputs
        validateInputs(x, y);

        // Calculate wirelength
        DTYPE wirelength = flute_wl(x.size(), x.data(), y.data(), FLUTE_ACCURACY);
        std::cout << "Total wirelength: " << wirelength << std::endl;

        // Get the routing tree
        Tree tree = flute(x.size(), x.data(), y.data(), FLUTE_ACCURACY);
        
        // Verify tree properties
        if (tree.deg != x.size()) {
            throw FluteException("Tree degree mismatch with input size");
        }
        if (tree.length <= 0) {
            throw FluteException("Invalid tree length");
        }

        std::cout << "Tree degree: " << tree.deg << std::endl;
        std::cout << "Tree length: " << tree.length << std::endl;

        // Write tree to SVG file
        try {
            write_svg(tree, "test_cpp_routing_tree.svg");
            std::cout << "Saving tree to " << "test_cpp_routing_tree.svg"<< std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to write SVG file: " << e.what() << std::endl;
        }

        // Clean up tree (in case of exceptions)
        free_tree(tree);

        std::cout << "Cpp tests passed! " << std::endl;

        return 0;

    } catch (const FluteException& e) {
        std::cerr << "Flute Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Unexpected Error: " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "Unknown Error occurred" << std::endl;
        return 3;
    }
}