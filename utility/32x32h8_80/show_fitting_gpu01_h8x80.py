# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import tensorflow as tf
import os
import time
NH = 80
IN_NODE = 2
H1_NODE = NH
H2_NODE = NH
H3_NODE = NH
H4_NODE = NH
H5_NODE = NH
H6_NODE = NH
H7_NODE = NH
H8_NODE = NH
OUT_NODE = 1
MODEL_SAVE_PATH = './model/'
MODEL_NAME = '8h_33x33_model'
# LR = 0.001
#################
# Build network 
#################
Weights1 = tf.Variable(tf.random_normal([IN_NODE,H1_NODE]))
biases1 = tf.Variable(tf.zeros([1,H1_NODE]) + 0.1) 

Weights2 = tf.Variable(tf.random_normal([H1_NODE,H2_NODE]))
biases2 = tf.Variable(tf.zeros([1,H2_NODE]) + 0.1) 

Weights3 = tf.Variable(tf.random_normal([H2_NODE,H3_NODE]))
biases3 = tf.Variable(tf.zeros([1,H3_NODE]) + 0.1) 

Weights4 = tf.Variable(tf.random_normal([H3_NODE,H4_NODE]))
biases4 = tf.Variable(tf.zeros([1,H4_NODE]) + 0.1) 


Weights5 = tf.Variable(tf.random_normal([H4_NODE,H5_NODE]))
biases5 = tf.Variable(tf.zeros([1,H5_NODE]) + 0.1) 

Weights6 = tf.Variable(tf.random_normal([H5_NODE,H6_NODE]))
biases6 = tf.Variable(tf.zeros([1,H6_NODE]) + 0.1) 

Weights7 = tf.Variable(tf.random_normal([H6_NODE,H7_NODE]))
biases7 = tf.Variable(tf.zeros([1,H7_NODE]) + 0.1) 

Weights8 = tf.Variable(tf.random_normal([H7_NODE,H8_NODE]))
biases8 = tf.Variable(tf.zeros([1,H8_NODE]) + 0.1) 

Weights9 = tf.Variable(tf.random_normal([H8_NODE,OUT_NODE]))
biases9 = tf.Variable(tf.zeros([1,OUT_NODE]) + 0.1) 
# def add_layer(inputs, in_size, out_size, actication_function = None):
#     Weights = tf.Variable(tf.random_normal([in_size,out_size]))
#     biases = tf.Variable(tf.zeros([1,out_size]) + 0.1) 
#     Wx_plus_b = tf.matmul(inputs, Weights) + biases
    
#     if actication_function is None:
#         outputs = Wx_plus_b
#     else:
#         outputs = actication_function(Wx_plus_b)

#     return outputs


def forward(t,x):
    # global Weights1, biases1, Weights2, biases2
    # Combine the two
    net_in = tf.concat([t,x],1)
    # h1 = add_layer(net_in, IN_NODE, H1_NODE, actication_function = tf.nn.sigmoid)
    # net_out = add_layer(h1, H1_NODE, OUT_NODE, actication_function = None)
    # return net_out
    Wx_plus_b = tf.matmul(net_in, Weights1) + biases1
    h1 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h1, Weights2) + biases2
    h2 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h2, Weights3) + biases3
    h3 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h3, Weights4) + biases4
    h4 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h4, Weights5) + biases5
    h5 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h5, Weights6) + biases6
    h6 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h6, Weights7) + biases7
    h7 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h7, Weights8) + biases8
    h8 = tf.nn.sigmoid(Wx_plus_b)

    Wx_plus_b = tf.matmul(h8, Weights9) + biases9

    net_out = Wx_plus_b

    return net_out


#################
# Prepare data
#################
NT = 33
NX = 33
# t_data
data = np.loadtxt('dat32x32.txt')
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
loss = tf.reduce_mean(tf.square(us - net_out))

########################
# Resrore NN
########################
def pull_model():
    # cmd = 'scp -r  jack@10.143.7.153:~/jack/github/use-tensorflow-to-solve-ode-pde/utility/model .'
    # cmd = 'scp -r  jack@10.143.7.153:~/model .'
    cmd = 'scp -r  jack@10.143.7.153:~/jack/github/use-tensorflow-to-solve-ode-pde/32x32h8_80/model .'
    os.system(cmd)

init = tf.global_variables_initializer()
# sess = tf.Session()
saver = tf.train.Saver()




fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')  
T,X = np.meshgrid(np.linspace(0,1,NT),np.linspace(0,1,NX))
plt.ion()

with tf.Session() as sess:
    sess.run(init)
    for i in range (4000000000):
        pull_model()
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored")
        print(sess.run(loss, feed_dict={ts:t_data, xs:x_data, us:u_data}))
        U = sess.run(net_out, feed_dict={ts:t_data, xs:x_data})
        u_p_data = U
        U = U.reshape(NX,NT,order='C')
        try:
            surfaces.remove()
        except Exception:
            pass
        surfaces = ax.plot_surface(T,X,U,rstride=1,cstride=1,cmap=plt.cm.jet)
	
        h4_200_fiting_data = np.append(t_data, x_data, axis=1)
        h4_200_fiting_data = np.append(h4_200_fiting_data ,u_p_data ,axis=1)
        np.savetxt('h8_80_fiting_data.txt', h4_200_fiting_data, fmt='%f')
        # Weights1_s = Weights1.eval()
        # biases1_s = biases1.eval()
        # Weights2_s = Weights2.eval()
        # biases2_s = biases2.eval()

        # print(Weights1_s)
        # print(biases1_s)

        # print(Weights2_s)
        # print(biases2_s)

        # np.savetxt(MODEL_SAVE_PATH+'Weights1_s.txt',Weights1_s,fmt='%f')
        # np.savetxt(MODEL_SAVE_PATH+'biases1_s.txt',biases1_s,fmt='%f')
        # np.savetxt(MODEL_SAVE_PATH+'Weights2_s.txt',Weights2_s,fmt='%f')
        # np.savetxt(MODEL_SAVE_PATH+'biases2_s.txt',biases2_s,fmt='%f')

        plt.pause(5)
              

