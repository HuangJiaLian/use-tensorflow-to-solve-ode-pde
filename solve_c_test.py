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
MODEL_NAME = 'h200_big_surface_model_c'
LR = 0.001



#################
# Build network
#################
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

def forward(t,x):
    # Combine the two
    net_in = tf.concat([t,x],1)
    h1 = add_layer(net_in, IN_NODE, H1_NODE, actication_function = tf.nn.sigmoid)
    net_out = add_layer(h1, H1_NODE, OUT_NODE, actication_function = None)
    return net_out

#################
# Prepare data
#################
NT = 401
NX = 33
# t_data
data = np.loadtxt('solSinEF7_big.txt')
temp = data[:,0]
t_data = temp[:,np.newaxis]
print(t_data, t_data.shape)

# x_data
temp = np.linspace(0,1,NX)
temp = temp[:,np.newaxis]
x_data = np.repeat(temp,NT,axis=0)
print(x_data, x_data.shape)



ts = tf.placeholder(tf.float32,shape=(None, 1))
xs = tf.placeholder(tf.float32,shape=(None, 1))



########################
# Define loss function
########################
# c = 7
# pi = math.pi
#
net_out = forward(ts,xs)
# U = net_out
# Ut = tf.gradients(U,ts)[0]
# Ux = tf.gradients(U,xs)[0]
# Uxx = tf.gradients(Ux,xs)[0]
# temp0 = c*tf.sin(2*pi*xs)
# s_term = tf.multiply(temp0,U)
# SSEu = tf.reduce_mean(tf.reduce_sum(tf.square(Ut - Uxx - s_term)))
#
#
# zeros = np.zeros([NT*NX,1])
# ones = np.ones([NT*NX,1])
# E = forward(zeros, ts)
# F = forward(ones, ts)
# G = forward(xs,zeros)
# SSEb = (tf.reduce_mean(tf.reduce_sum(tf.square(E-F)))) + (tf.reduce_mean(tf.reduce_sum(tf.square(G-ones))))
#
# loss = SSEu + SSEb
# train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)


########################
# train NN
########################
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')
T,X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0,1,NX))
plt.ion()

for i in range(4000000000):
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored")
    # if i % 100 == 0:
    #     print(sess.run(loss, feed_dict={ts: t_data, xs: x_data}))
    U = sess.run(net_out, feed_dict={ts:t_data, xs:x_data})
    U = U.reshape(NX,NT,order='C')
    try:
        surfaces.remove()
    except Exception:
        pass
    surfaces = ax.plot_surface(T,X,U,rstride=1,cstride=1,cmap=plt.cm.jet)
    plt.pause(5)

'''
t_delta = 1/NT
a1 = 1.
b1 = 70.
b2 = 70.
c = 7
pi = math.pi

# In[3]:


# loss function
U = forward(xs, ts)
Ux = tf.gradients(U,xs)[0]
Uxx = tf.gradients(Ux,xs)[0]
Ut = tf.gradients(U,ts)[0]


temp0 = tf.sin(2*pi*xs)
D = tf.multiply(c*U, temp0)
SSEu = a1*tf.reduce_mean(tf.reduce_sum(tf.square(Ut - Uxx - D)))
# SSEu = (a1/(NT*NX))*tf.reduce_sum(tf.square(Ut - Uxx - D))

zeros = np.zeros([NT*NT,1])
ones = np.ones([NT*NT,1])

E = forward(zeros, ts)
F = forward(ones, ts)
G = forward(xs,zeros)
# SSEb = (b1/(NT + 1))*tf.reduce_sum(tf.square(E-F)) + (b2/(NX + 1))*tf.reduce_sum(tf.square(G-1.))
# SSEb = (b1/(NT))*tf.reduce_sum(tf.square(E-F)) + (b2/(NX))*tf.reduce_sum(tf.square(G-1.))
SSEb = (b1*tf.reduce_mean(tf.reduce_sum(tf.square(E-F)))) + (b2*tf.reduce_mean(tf.reduce_sum(tf.square(G-ones))))
loss = SSEu + SSEb

train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)


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

        try:
            surfaces.remove()
        except Exception:
            pass
        surfaces = ax.plot_surface(T,X,Z,rstride=1,cstride=1,cmap=plt.cm.jet)
        plt.pause(0.01)
        
'''
