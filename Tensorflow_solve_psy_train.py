
# coding: utf-8



import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import os
from numpy import random

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'solve_equation_model'

# Initial Condition
C1 = 1

def add_layer(inputs, in_size, out_size, actication_function = None):
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # Because the recommend initial
                                                       # value of biases != 0; so add 0.1
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
    if actication_function is None:
        outputs = Wx_plus_b
    else:
        outputs = actication_function(Wx_plus_b)
    return outputs

# Left part of initial equation
def A(x):
    return x + (1. + 3.*x**2) / (1. + x + x**3)


# Right part of initial equation
def B(x):
    return x**3 + 2.*x + x**2 * ((1. + 3.*x**2) / (1. + x + x**3))

# d(psy)/dx = f(x, psy)
# This is f() function on the right
def f(x, psy):
    return B(x) - psy * A(x)

# Analytical solution of current problem
def psy_analytic(x):
    return (C1*np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2


# 0 到 1 中间取 nx 个点
nx = 50

x_space = np.linspace(0, 1, nx)    
y_space = psy_analytic(x_space)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_space, y_space)
plt.ion()
plt.ylim(-0.5,1.5)
plt.show()


# create  training data
x_space = np.linspace(0,1,nx)[:, np.newaxis] 
xs = tf.placeholder(tf.float32, [None, 1])  # * rows, 1 col


# define hidden layer and output layer
l1 = add_layer(xs, 1, 10, actication_function = tf.nn.sigmoid)
# l2 = add_layer(l1, 50, 10, actication_function = tf.nn.sigmoid)
# # h_fc2_drop = tf.nn.dropout(l2,KEEP)
# l3 = add_layer(l2, 10, 50, actication_function = tf.nn.sigmoid)
# l4 = add_layer(l3, 50, 10, actication_function = tf.nn.sigmoid)
# l5 = add_layer(l4, 10, 50, actication_function = tf.nn.sigmoid)
# l6 = add_layer(l5, 50, 50, actication_function = tf.nn.sigmoid)
# l7 = add_layer(l6, 50, 50, actication_function = tf.nn.sigmoid)
# l8 = add_layer(l7, 50, 50, actication_function = tf.nn.sigmoid)
# l9 = add_layer(l8, 50, 50, actication_function = tf.nn.sigmoid)
# l10 = add_layer(l9, 50, 50, actication_function = tf.nn.sigmoid)
net_out = add_layer(l1, 10, 1, actication_function = None)


d_neural_network_dx = tf.gradients(ys=net_out, xs=xs)
pys_t = C1 + xs * net_out
# pys_t =  xs * net_out
# d_psy_t = net_out + xs * d_neural_network_dx # The same with the following line
d_psy_t = tf.gradients(ys=pys_t, xs=xs)
Bx = xs**3 + 2*xs + (xs**2)*(1+3*(xs**2))/(1 + xs + xs**3)
Ax = xs + (1+3*( xs**2 ))/( 1+ xs + xs**3 )
func = Bx - net_out*Ax


loss = tf.reduce_mean(tf.reduce_sum(tf.square(d_psy_t - func), 
                reduction_indices=[1]))

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(d_psy_t - func), 
#                 reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(1e-1).minimize(loss)
saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
# ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
# if ckpt and ckpt.model_checkpoint_path:
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     print("Model restored")
    
for i in range (4000000):
    # ??? tensorflow know update the x_data, y_data at each train ???
    # data = random.rand(1, 10)
    sess.run(train_step, feed_dict={xs:x_space})
    if i % 100 == 0:
        print(sess.run(loss,feed_dict={xs:x_space}))
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        # Important 
        res = sess.run(pys_t, feed_dict={xs:x_space})
        try:
            ax.lines.remove(lines[0])
        except Exception :
            pass
        lines = ax.plot(x_space, res, 'r-', lw = 1)
        plt.pause(0.01)
