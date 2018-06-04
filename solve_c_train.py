# coding: utf-8
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import math 
import os

IN_NODE = 2
H1_NODE = 200
OUT_NODE = 1
MODEL_SAVE_PATH = './model/'
# MODEL_NAME = 'diffusion_model_c'
MODEL_NAME = 'h200_big_surface_model'
LR = 0.0001
LOAD_FILE = 'solSinEF7_big.txt'


def ooops():
    print('ooops')

#################
# Build network
#################
try:
    temp = np.loadtxt(MODEL_SAVE_PATH+'Weights1_s.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([IN_NODE,H1_NODE])
    # print(temp,temp.shape)
    Weights1 = tf.Variable(temp)
    print('Well Done')    
except Exception as e:
    ooops()
    Weights1 = tf.Variable(tf.random_normal([IN_NODE,H1_NODE]))

try:
    temp = np.loadtxt(MODEL_SAVE_PATH+'biases1_s.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([1,H1_NODE])
    # print(temp,temp.shape)
    biases1 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    biases1 = tf.Variable(tf.zeros([1,H1_NODE]) + 0.1) 

try:
    temp = np.loadtxt(MODEL_SAVE_PATH+'Weights2_s.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([H1_NODE,OUT_NODE])
    # print(temp,temp.shape)
    Weights2 = tf.Variable(temp)
    print('Well Done')    
except Exception as e:
    ooops()
    Weights2 = tf.Variable(tf.random_normal([H1_NODE,OUT_NODE]))

try:
    temp = np.loadtxt(MODEL_SAVE_PATH+'biases2_s.txt', dtype=np.float32)
    # It's special because temp is number, which is diffenrent 
    # from above
    temp = temp.reshape([1,1])
    # print(temp)
    biases2 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    biases2 = tf.Variable(tf.zeros([1,OUT_NODE]) + 0.1)

# def add_layer(inputs, in_size, out_size, actication_function = None):
#     Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#     biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) # Because the recommend initial
#                                                        # value of biases != 0; so add 0.1
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
#     if actication_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = actication_function(Wx_plus_b)
#     return outputs

# def forward(t,x):
#     # Combine the two
#     net_in = tf.concat([t,x],1)
#     h1 = add_layer(net_in, IN_NODE, H1_NODE, actication_function = tf.nn.sigmoid)
#     net_out = add_layer(h1, H1_NODE, OUT_NODE, actication_function = None)
#     return net_out

def forward(t,x):
    global Weights1, biases1, Weights2, biases2
    # Combine the two
    net_in = tf.concat([t,x],1)
    Wx_plus_b = tf.matmul(net_in, Weights1) + biases1
    h1 = tf.nn.sigmoid(Wx_plus_b)
    Wx_plus_b = tf.matmul(h1, Weights2) + biases2
    net_out = Wx_plus_b
    return net_out

#################
# Prepare data
#################
NT = 401
NX = 33
# t_data
data = np.loadtxt(LOAD_FILE)
temp = data[:,0]
t_data = temp[:,np.newaxis]
# print(t_data, t_data.shape)

# x_data
temp = np.linspace(0,1,NX)
temp = temp[:,np.newaxis]
x_data = np.repeat(temp,NT,axis=0)
# print(x_data, x_data.shape)



ts = tf.placeholder(tf.float32,shape=(None, 1))
xs = tf.placeholder(tf.float32,shape=(None, 1))



########################
# Define loss function
########################
c = 7
pi = math.pi

net_out = forward(ts,xs)

U = net_out
Ut = tf.gradients(U,ts)[0]
Ux = tf.gradients(U,xs)[0]
Uxx = tf.gradients(Ux,xs)[0]
temp0 = c*tf.sin(2*pi*xs)
s_term = tf.multiply(temp0,U)
# SSEu = tf.reduce_mean(tf.reduce_sum(tf.square(Ut - Uxx - s_term)))
SSEu = tf.reduce_mean(tf.square(Ut - Uxx - s_term))


zeros = np.zeros([NT*NX,1])
ones = np.ones([NT*NX,1])
E = forward(zeros, ts)
F = forward(ones, ts)
G = forward(xs,zeros)
SSEb = (tf.reduce_mean(tf.square(E-F))) + (tf.reduce_mean(tf.square(G-ones)))

loss = SSEu + SSEb
train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)


########################
# train NN
########################
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")

for i in range(4000000000):
    sess.run(train_step, feed_dict={ts: t_data, xs: x_data})
    if i % 100 == 0:
        print(sess.run(loss, feed_dict={ts: t_data, xs: x_data}))
    if i % 1000 == 0:
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
