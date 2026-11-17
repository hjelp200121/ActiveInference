import numpy as np

A = np.array([0,0.2,0.4,1])
A = np.clip(A, 0, 0.9)
print(A)