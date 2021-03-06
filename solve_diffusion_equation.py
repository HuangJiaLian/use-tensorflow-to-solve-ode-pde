# coding: utf-8
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
from numpy import random
import math 
import os

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'diffusion_model'

INPUT_NODE = 2
LAYER1_NODE = 50
LAYER2_NODE = 50
LAYER3_NODE = 50
LAYER4_NODE = 50
LAYER5_NODE = 50
LAYER6_NODE = 50
LAYER7_NODE = 50
LAYER8_NODE = 50
LAYER9_NODE = 50
LAYER10_NODE = 50
OUTPUT_NODE = 1

# Number of x and t
NT = 401
NX = 33
LR = 1e-4
# LR = 0.005

t_delta = 1/(NT-1)
a1 = 1.
b1 = 70.
b2 = 70.
c = 1
pi = math.pi



# Build network 
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


def forward(x,t):
    net_in = tf.concat([x,t],1)
    print(type(net_in))
    h1 = add_layer(net_in, INPUT_NODE, LAYER1_NODE, actication_function = tf.nn.sigmoid)
    h2 = add_layer(h1, LAYER1_NODE, LAYER2_NODE, actication_function = tf.nn.sigmoid)
    h3 = add_layer(h2, LAYER2_NODE, LAYER3_NODE, actication_function = tf.nn.sigmoid)
    h4 = add_layer(h3, LAYER3_NODE, LAYER4_NODE, actication_function = tf.nn.sigmoid)
    h5 = add_layer(h4, LAYER4_NODE, LAYER5_NODE, actication_function = tf.nn.sigmoid)
    h6 = add_layer(h5, LAYER5_NODE, LAYER6_NODE, actication_function = tf.nn.sigmoid)
    h7 = add_layer(h6, LAYER6_NODE, LAYER7_NODE, actication_function = tf.nn.sigmoid)
    h8 = add_layer(h7, LAYER7_NODE, LAYER8_NODE, actication_function = tf.nn.sigmoid)
    h9 = add_layer(h8, LAYER8_NODE, LAYER9_NODE, actication_function = tf.nn.sigmoid)
    h10 = add_layer(h9, LAYER9_NODE, LAYER10_NODE, actication_function = tf.nn.sigmoid)
    net_out = add_layer(h10, LAYER10_NODE, OUTPUT_NODE, actication_function = None)
    return net_out

xs = tf.placeholder(tf.float32,shape=(None, 1))
ts = tf.placeholder(tf.float32,shape=(None, 1))
ts_2 = ts + t_delta


# In[3]:


# loss function
U = forward(xs, ts)

A = forward(xs, ts_2)
B = U

Ax = tf.gradients(A,xs)[0]
Axx = tf.gradients(Ax,xs)[0]
Bx = tf.gradients(B,xs)[0]
Bxx = tf.gradients(Bx,xs)[0]
C = (t_delta/2.0)*(Axx + Bxx)

temp0 = tf.sin(2*pi*xs)
D = tf.multiply((t_delta/2.0)*c*(A + B), temp0)

SSEu= a1 * tf.reduce_mean(tf.reduce_sum(tf.square(A-B-C-D)))

zeros = np.zeros([NT*NX,1])
ones = np.ones([NT*NX,1])

E = forward(zeros, ts)
F = forward(ones, ts)
G = forward(xs,zeros)


SSEb = (b1*tf.reduce_mean(tf.reduce_sum(tf.square(E-F)))) + (b2*tf.reduce_mean(tf.reduce_sum(tf.square(G-ones))))

loss = SSEu + SSEb

train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# In[4]:


# Trainning Data
arr = []
for x in range(NX):
    for t in range(NT):
        b = [x/(NX-1), t/(NT-1)]
        arr.append(b)
#         t += 1
data = np.array(arr)
# print(data)
t = data[:,1][:, np.newaxis]
x = data[:,0][:, np.newaxis]
# print(t,t.shape)
# print(x,x.shape)
# tt = np.linspace(0,1,NT)
# xt = np.linspace(0,1,NX)
# T, X = np.meshgrid(tt,xt)
# T.reshape(-1,1)
# X.reshape(-1,1)


# In[5]:


# saver 用来保存训练模型
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 断点续训的功能
# 非常实用啊
ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")

#设置三维坐标  
fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')  
# x_space =  np.arange(0, 1, 1/NX)
# t_space =  np.arange(0, 1, 1/NT)

T,X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0,1,NX))
print(T.shape)
print(X.shape)
plt.ion()

for i in range(400000000000):
    sess.run(train_step, feed_dict={xs:x, ts:t})    
    if i%100 == 0:
        print(sess.run(loss, feed_dict={xs:x, ts:t}))
        
        # 保存训练模型
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
                
        Z = sess.run(U, feed_dict={xs:x, ts:t})
        Z = Z.reshape(NX,NT,order='C')

        # tempx = X.reshape(-1,1,order='F')
        # tempt = T.reshape(-1,1,order='C')
        # tempx = X.reshape(-1,1,order='F')
        # tempt = T.reshape(-1,1,order='F')
        # print(tempx, tempx.shape)
        # print(tempt, tempt.shape)
        # tempxt = np.append(tempx,tempt, axis = 1)
        # print(tempxt) 
        # print(T.shape)
        # print(X.shape)
        # print(Z.shape)
        try:
            surfaces.remove()
        except Exception:
            pass
        surfaces = ax.plot_surface(T,X,Z,rstride=1,cstride=1,cmap=plt.cm.jet)
        plt.pause(0.01)