
# coding: utf-8

# In[7]:


import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import os

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'solve_equation_model'


# In[8]:


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

# def add_layer(inputs, in_size, out_size, actication_function = None):
#     Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#     Wx_plus_b = tf.matmul(inputs, Weights)

#     if actication_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = actication_function(Wx_plus_b)
#     return outputs

# In[9]:


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
    return (1*np.exp((-x**2)/2.)) / (1. + x + x**3) + x**2


# In[10]:


# 0 到 1 中间取 10 个点
nx = 20


# In[11]:


# 这个是标准答案
x_space = np.linspace(0, 1, nx)    
y_space = psy_analytic(x_space)

# plt.figure()
# plt.plot(x_space, y_space)
# plt.ion()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.scatter(x_data,y_data)
# plt.ion()
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x_space, y_space)
plt.ion()
plt.show()

# In[15]:


# create  training data
x_space = np.linspace(0,1,nx)[:, np.newaxis] 
xs = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col


# define hidden layer and output layer
# l1 = add_layer(xs, 1, 10, actication_function = tf.nn.sigmoid)
# prediction = add_layer(l1, 10, 1, actication_function = None)

# define hidden layer and output layer
l1 = add_layer(xs, 1, 16, actication_function = tf.nn.sigmoid)
# l2 = add_layer(l1, 50, 50, actication_function = tf.nn.sigmoid)
prediction = add_layer(l1, 16, 1, actication_function = None)


d_neural_network_dx = tf.gradients(ys=prediction, xs=xs)
# d_psy_t = prediction + xs * d_neural_network_dx
Bx = xs**3 + 2*xs + (xs**2)*(1+3*(xs**2))/(1 + xs + xs**3)
Ax = xs + (1+3*( xs**2 ))/( 1+ xs + xs**3 )
func = Bx - prediction*Ax


loss = tf.reduce_mean(tf.reduce_sum(tf.square(d_neural_network_dx - func), 
                reduction_indices=[1]))

# loss = tf.reduce_mean(tf.reduce_sum(tf.square(d_psy_t - func), 
#                 reduction_indices=[1]))

# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(1e-1).minimize(loss)
saver = tf.train.Saver()

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")
    
for i in range (4000000):
    # ??? tensorflow know update the x_data, y_data at each train ???
    sess.run(train_step, feed_dict={xs:x_space})
    if i % 100 == 0:
        print(sess.run(loss,feed_dict={xs:x_space}))
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
        res = sess.run(prediction, feed_dict={xs:x_space})
        try:
            ax.lines.remove(lines[0])
        except Exception :
            pass
        lines = ax.plot(x_space, res, 'r-', lw = 1)
        plt.pause(0.01)
