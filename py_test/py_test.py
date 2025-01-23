# import numpy as np
# import pyflute

# # Initialize FLUTE
# pyflute.readLUT()

# try:
#     # Example with 4 pins
#     x = np.array([0, 100, 100, 0], dtype=np.int32)
#     y = np.array([0, 0, 100, 100], dtype=np.int32)

#     # Calculate wirelength
#     wl = pyflute.flute_wl(x, y, accuracy=10)
#     print(f"Wirelength: {wl}")

#     # Get routing tree
#     tree = pyflute.flute(x, y, accuracy=10)
#     print(f"Tree degree: {tree.deg}")
#     print(f"Tree length: {tree.length}")

#     # Access branches
#     for i, branch in enumerate(tree.branches):
#         print(f"Branch {i}: ({branch.x}, {branch.y}) -> neighbor {branch.n}")

#     # Free tree memory
#     pyflute.free_tree(tree)

# finally:
#     # Clean up
#     pyflute.deleteLUT()
import numpy as np
import pyflute

# Initialize FLUTE
pyflute.readLUT()

# Create input arrays
x = np.array([0, 100, 100, 0], dtype=np.int32)
y = np.array([0, 0, 100, 100], dtype=np.int32)

# Calculate wirelength
wl = pyflute.flute_wl(x, y, accuracy=10)
print(f"Wirelength: {wl}")

# Get routing tree
tree = pyflute.flute(x, y, accuracy=10)
print(f"Tree length: {tree.length}")

# Cleanup
pyflute.deleteLUT()