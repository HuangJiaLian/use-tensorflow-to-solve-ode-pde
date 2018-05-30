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
# 32 32 
NX = 32
NT = 32
LR = 1e-1


t_delta = 1/NT
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

temp = tf.reduce_sum(tf.square(A-B-C-D))
SSEu= a1 * tf.reduce_mean(temp)
# SSEu= (a1/(NT*(NX+1))) * temp

zeros = np.zeros([NT*NT,1])
ones = np.ones([NT*NT,1])

E = forward(zeros, ts)
F = forward(ones, ts)
G = forward(xs,zeros)


SSEb = (b1*tf.reduce_mean(tf.reduce_sum(tf.square(E-F)))) + (b2*tf.reduce_mean(tf.reduce_sum(tf.square(G-ones))))
# SSEb = (b1/(NT + 1))*tf.reduce_sum(tf.square(E-F)) + (b2/(NX + 1))*tf.reduce_sum(tf.square(G-1.))

loss = SSEu + SSEb

train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)


# In[4]:


# Trainning Data
arr = []
for x in range(NX):
    for t in range(NT):
        b = [x/NX, t/NT]
        arr.append(b)
#         t += 1
data = np.array(arr)
# print(data)
x = data[:,0][:, np.newaxis]
t = data[:,1][:, np.newaxis]
print(x)



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
ax.set_xlabel('X')
ax.set_ylabel('T')
ax.set_zlabel('U')  
x_space =  np.arange(0, 1, 1/NX)
t_space =  np.arange(0, 1, 1/NT)
X,T = np.meshgrid(x_space,t_space)
# print(X.shape)
# print(T.shape)
plt.ion()

for i in range(400000000000):
    sess.run(train_step, feed_dict={xs:x, ts:t})    
    if i%1000 == 0:
        print(sess.run(loss, feed_dict={xs:x, ts:t}))
        
        # 保存训练模型
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME))
                
        Z = sess.run(U, feed_dict={xs:X.reshape(-1,1,order='F'), ts:T.reshape(-1,1,order='C')})
        Z = Z.reshape(NX,NT,order='C')
        # print(Z.shape)
        try:
            surfaces.remove()
        except Exception:
            pass
        surfaces = ax.plot_surface(X,T,Z,rstride=1,cstride=1,cmap=plt.cm.jet)
        plt.pause(0.01)