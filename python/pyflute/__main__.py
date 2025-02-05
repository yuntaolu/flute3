# python/pyflute/__main__.py
import numpy as np
from . import readLUT, deleteLUT, flute, flute_wl

def main():
    readLUT()
    try:
        x = np.array([0, 100, 100, 0], dtype=np.int32)
        y = np.array([0, 0, 100, 100], dtype=np.int32)
        
        wl = flute_wl(x, y)
        print(f"Wirelength: {wl}")
        
        tree = flute(x, y)
        print(f"Tree: {tree}")
    finally:
        deleteLUT()

if __name__ == "__main__":
    main()