import numpy as np
import matplotlib.pyplot as plt
m = 2
n = 30
iters = 5
w_star = np.random.uniform(0,1,3)
w_star[0] = 0.1
x_train = np.random.uniform(-1,1,(n,m))
y_train = np.sign(np.dot(x_train,w_star[1:])+ w_star[0]) 

w = np.random.uniform(0,1,m+1)
x = np.linspace(-1,1,100)
for j in range(iters):
    for i in range(n):
        if np.sign(np.dot(x_train[i], w[1:]) + w[0]) != y_train[i]:
            w[1:] = w[1:] + y_train[i] * x_train[i]
            w[0] = w[0] + y_train[i] 

    plt.scatter(x_train[np.where(y_train == 1)][:,0],x_train[np.where(y_train == 1)][:,1])
    plt.scatter(x_train[np.where(y_train == -1)][:,0],x_train[np.where(y_train == -1)][:,1])
    y_gt = (-(w_star[0] / w_star[2]) / (w_star[0] / w_star[1]))* x + (-w_star[0] / w_star[2])
    y_h = (-(w[0] / w[2]) / (w[0] / w[1]))* x + (-w[0] / w[2])
    plt.plot(x,y_gt, c = 'purple')
    plt.plot(x, y_h, c = 'red')
    plt.show()