import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# Data
A = [2, 5, 9, 11, 15, 18, 25, 22, 27, 29, 33, 33, 38, 40, 42]
b = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
fig1 = plt.figure("Gradient Descent for Linear Regression")
ax = plt. axes(xlim=(-10, 55), ylim=(-1, 18))
plt.plot(A, b, 'ro')
A = np.array([A]).T
b = np.array([b]).T
# print(A)
# print(b)
lr = linear_model.LinearRegression()
lr.fit(A, b)
x0_gd = np.linspace(1, 42, 2)
y0_sklearn = lr.intercept_[0] + lr.coef_[0][0]*x0_gd
plt.plot(x0_gd, y0_sklearn ,color='green')

ones = np.ones((A.shape[0], 1), dtype = np.int8)
A = np.concatenate((ones, A), axis=1)
print(A)

# Random initial line
x_init = np.array([[1], [2]])
y0_init = x_init[0][0] + x_init[1][0] * x0_gd
plt.plot(x0_gd, y0_init, color='yellow')

# Build GD-Algorithms
# 1. f(x) = |Ax-b|^2 <=> cost(x) 
# 2. f'(x) = 2A^T|Ax-b| grad(x)
# 3. Gradient-Descent()

def cost(x):
    m = A.shape[0]
    return 0.5/m * np.linalb.norm(A.dot(x) - b, 2)**2

def grad(x):
    m = A.shape[0]
    return 1/m * A.T.dot(A.dot(x)-b)

def gradient_descent(x_init, lr, iter):
    x_list = [x_init]
    for i in range(iter):
        x_new = x_list[-1] - lr * grad(x_list[-1])
        x_list.append(x_new)

    return x_list

iter = 90
lr = 1e-4

x_list = gradient_descent(x_init, lr, iter)

for i in range(len(x_list)):
    y0_x_list = x_list[i][0] + x_list[i][1] * x0_gd
    plt.plot(x0_gd, y0_x_list, color = 'yellow')
plt.show()