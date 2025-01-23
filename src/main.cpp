// src/main.cpp
#include <iostream>
#include <flute.h>  // Include the Flute header

using namespace Flute;

int main() {
    // Initialize Flute lookup tables
    readLUT();

    // Example: Calculate wirelength for 12 pins
    int numPins = 12;
    DTYPE x[] = {0, 100, 100, 0, 0, 80, 50, 0, 20, 10, 80, 50};    // x coordinates
    DTYPE y[] = {0, 0, 100, 100, 0, 10, 80, 50, 10, 80, 50, 80};    // y coordinates

    // Calculate wirelength with maximum accuracy (default = 24)
    DTYPE wirelength = flute_wl(numPins, x, y, FLUTE_ACCURACY);
    std::cout << "Total wirelength: " << wirelength << std::endl;

    // Get the actual routing tree
    Tree tree = flute(numPins, x, y, FLUTE_ACCURACY);
    std::cout << "Tree degree: " << tree.deg << std::endl;
    std::cout << "Tree length: " << tree.length << std::endl;

    // Write tree to SVG file for visualization
    write_svg(tree, "routing_tree.svg");

    // Clean up
    free_tree(tree);
    deleteLUT();

    return 0;
}