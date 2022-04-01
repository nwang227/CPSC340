import numpy as np

X = np.array([0, -3, 1])

W = np.array([0.5, 1, 0.5])

Z = np.linalg.solve(W @ W.T, W @ X.T)