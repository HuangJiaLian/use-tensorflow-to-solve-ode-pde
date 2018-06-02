
# coding: utf-8



import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import tensorflow as tf
import os
import time

IN_NODE = 2
H1_NODE = 50
OUT_NODE = 1
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'surface_model'
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
# print(t_data, t_data.shape)

# x_data
temp = np.linspace(0,1,NX)
temp = temp[:,np.newaxis]
x_data = np.repeat(temp,NT,axis=0)
# print(x_data, x_data.shape)


# u_data
temp = data[:,2]
u_data = temp[:,np.newaxis]
# print(u_data, u_data.shape)


ts = tf.placeholder(tf.float32,shape=(None, 1))
xs = tf.placeholder(tf.float32,shape=(None, 1))
# us: The standard answer 
us = tf.placeholder(tf.float32,shape=(None, 1))

########################
# Define loss function
########################
net_out = forward(ts, xs)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(us - net_out)))

########################
# Resrore NN
########################
def pull_model():
    cmd = 'scp -r  jack@10.143.7.153:~/jack/github/use-tensorflow-to-solve-ode-pde/utility/model .'
    os.system(cmd)
    # print(cmd)

init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model restored")

fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')  
T,X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0,1,NX))
plt.ion()


for i in range (4000000000):
    pull_model()
    ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored")
    print(sess.run(loss, feed_dict={ts:t_data, xs:x_data, us:u_data}))
    U = sess.run(net_out, feed_dict={ts:t_data, xs:x_data})
    U = U.reshape(NX,NT,order='C')
    try:
        surfaces.remove()
    except Exception:
        pass
    surfaces = ax.plot_surface(T,X,U,rstride=1,cstride=1,cmap=plt.cm.jet)
    plt.pause(5)
          

