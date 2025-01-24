#!/usr/bin/env python3
import sys
import numpy as np
import pyflute

def test_flute():
    # Initialize
    pyflute.readLUT()

    try:
        # Test data
        x = np.array([0, 100, 100, 0, 50, 80, 50, 0, 20, 10, 80, 60], dtype=np.int32)
        y = np.array([0, 0, 100, 100, 90, 10, 80, 50, 10, 80, 50, 80], dtype=np.int32)

        # Test wirelength calculation
        wl = pyflute.flute_wl(x, y)
        print(f"Wirelength: {wl}")
        assert wl > 0, "Wirelength should be positive"
        
        # Test tree degree
        tree = pyflute.flute(x, y)
        print(f"Tree degree: {tree.deg}")
        assert tree.deg > 0, "Tree degree should be positive"
        
        # Test tree construction
        tree = pyflute.flute(x, y)
        print(f"Tree length: {tree.length}")
        assert tree.length == wl, "Tree length should match wirelength"

        # Test SVG output
        pyflute.write_svg(tree, "test_python_routing_tree.svg")

        print("Python tests passed!")
        return 0

    finally:
        # Cleanup
        pyflute.deleteLUT()

if __name__ == "__main__":
    sys.exit(test_flute())