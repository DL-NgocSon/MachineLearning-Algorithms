import math
import numpy as np

import matplotlib
import matplotlib.pyplot

def grad(x):
    return 2*x + 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD(lr, x0):
    x = [x0]
    for i in range(100):
        x_new = x[-1] - lr*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return x, i

x1, i1 = myGD(.1, -5)
x2, i2 = myGD(.1, 5)
print('Solution x = %f, cost = %f, iter %d' %(x1[-1], cost(x1[-1]), i1))
print('Solution x = %f, cost = %f, iter %d' %(x2[-1], cost(x2[-1]), i2))
