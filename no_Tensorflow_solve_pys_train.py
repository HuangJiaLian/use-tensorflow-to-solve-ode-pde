
# coding: utf-8

# ![微分方程](https://cdn-images-1.medium.com/max/800/1*OWbwgYIEU0dhVWmpug9QNw.png)
# 
# [参考链接](https://becominghuman.ai/neural-networks-for-solving-differential-equations-fa230ac5e04c)

# In[1]:


# https://becominghuman.ai/neural-networks-for-solving-differential-equations-fa230ac5e04c
# ODE: ordinary differential equation
# PDE: partial differential equation
# IC: initial condition
# BC: boundary condition

import autograd.numpy as np 
from autograd import grad
import autograd.numpy.random as npr
from autograd.core import primitive
from matplotlib import pyplot as plt
C1 = 1

# In[2]:


def A(x):
    '''
        Left part of initial equation
    '''
    return x + (1. + 3.*x**2) / (1. + x + x**3)


def B(x):
    '''
        Right part of initial equation
    '''
    return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

def f(x, psy):
    '''
        d(psy)/dx = f(x, psy)
        This is f() function on the right
    '''
    return B(x) - psy * A(x)


def psy_analytic(x):
    '''
        Analytical solution of current problem
    '''
    return (C1*np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2


# In[3]:


# 0 到 1 中间取 10 个点
nx = 10
dx = 1. / nx


# In[4]:


# 这个是标准答案
x_space = np.linspace(0, 1, nx)    
y_space = psy_analytic(x_space)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_space, y_space)
plt.ion()
plt.show()


# In[5]:


# sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# derivertive of sigmoid 
def sigmoid_grad(x):
    return sigmoid(x) * (1 - sigmoid(x))

# omit bias
# One hidden layer
def neural_network(W, x):
    a1 = sigmoid(np.dot(x, W[0]))
    return np.dot(a1, W[1])

# ？？？？
def d_neural_network_dx(W, x, k=1):
    return np.dot(np.dot(W[1].T, W[0].T**k), sigmoid_grad(x))

# 自定义的loss function
def loss_function(W, x):
    loss_sum = 0.
    for xi in x:
        net_out = neural_network(W, xi)[0][0]
        psy_t = C1 + xi * net_out
        d_net_out = d_neural_network_dx(W, xi)[0][0]
        d_psy_t = net_out + xi * d_net_out
        func = f(xi, psy_t)       
        err_sqr = (d_psy_t - func)**2
        loss_sum += err_sqr
    return loss_sum


# In[6]:


# 迭代1000次 mean squared error (MSE) of 0.0962
W = [npr.randn(1, 10), npr.randn(10, 1)]
lmb = 0.001

# x = np.array(1)
# print neural_network(W, x)
# print d_neural_network_dx(W, x)

for i in range(2000):
    loss_grad =  grad(loss_function)(W, x_space)
    
#     print loss_grad[0].shape, W[0].shape
#     print loss_grad[1].shape, W[1].shape
    
    W[0] = W[0] - lmb * loss_grad[0]
    W[1] = W[1] - lmb * loss_grad[1]
    if i % 100 == 0:
        print(loss_function(W, x_space))
        res = [1 + xi * neural_network(W, xi)[0][0] for xi in x_space]
        try:
            ax.lines.remove(lines[0])
        except Exception :
            pass
        lines = ax.plot(x_space, res, 'r-', lw = 1)
        plt.pause(0.01)
  #   	plt.figure()
		# plt.plot(x_space, y_space) 
		# plt.plot(x_space, res)
		# plt.show()
#     print (loss_function(W, x_space))


# In[7]:


# print(loss_function(W, x_space))
# res = [1 + xi * neural_network(W, xi)[0][0] for xi in x_space] 

# print(W)

plt.figure()
plt.plot(x_space, y_space) 
plt.plot(x_space, res)
plt.show()

