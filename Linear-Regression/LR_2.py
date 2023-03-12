import numpy as np
import matplotlib.pyplot as plt


b = [2, 5, 9, 11, 15, 18, 25, 22, 27, 29, 33, 33, 38, 39, 42, 42, 42, 39, 38, 30, 33, 28, 21, 25, 27, 18, 17, 11, 9, 4, 1]
A = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
plt.plot(A, b, 'ro')

A = np.array([A]).T
b = np.array([b]).T
print(A[:, 0])
A_square = np.array([A[:, 0]**2]).T
A = np.concatenate((A_square, A), axis=1)
ones = np.ones((A.shape[0], 1), dtype=np.int8)
print(ones)
print(A.shape)
A = np.concatenate((A, ones), axis=1)

x = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose()).dot(b)

x0 = np.linspace(1, 31, 10000)
y0 = x[0][0]*x0*x0 + x[1][0]*x0 + x[2][0]
# x_test  = np.array([[1, 31]]).T
# y_test = x_test*x[]
plt.plot(x0, y0)
plt.show()