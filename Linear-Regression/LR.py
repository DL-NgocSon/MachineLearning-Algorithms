import numpy as np
import matplotlib.pyplot as plt

# Data
A = [2, 5, 9, 11, 15, 18, 25, 22, 27, 29, 33, 33, 38, 40, 42]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
plt.plot(A, b, 'ro')

A = np.array([A]).T
b = np.array([b]).T
ones = np.ones((A.shape[0], 1), dtype=np.int8)
A = np.concatenate((A, ones), axis=1)
#print(A)
#(A^T*A)^-1 * A^T*b
x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)
print(x)

x0 = np.array([[1, 46]]).T
y0 = x[1][0] + x[0][0] * x0

plt.plot(x0, y0)
plt.show()