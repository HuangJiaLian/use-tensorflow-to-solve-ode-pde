# coding: utf-8
import numpy as np
import tensorflow as tf
import os
import time

IN_NODE = 2
H1_NODE = 200
OUT_NODE = 1
MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'h200_big_surface_model'
#LR = 1e-1
LR = 0.0004
LOAD_FILE = 'solSinEF7_big.txt'
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
data = np.loadtxt(LOAD_FILE)
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
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(us - net_out)))
loss = tf.reduce_mean(tf.square(us - net_out))
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdadeltaOptimizer(LR).minimize(loss)
# print(data)


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


for i in range (4000000000):
    sess.run(train_step, feed_dict={ts:t_data, xs:x_data, us:u_data})
    if i % 100 == 0:
        predict = sess.run(net_out, feed_dict={ts:t_data, xs:x_data, us:u_data})
        loss_value = sess.run(loss, feed_dict={ts:t_data, xs:x_data, us:u_data})
        # print(sess.run(loss, feed_dict={ts:t_data, xs:x_data, us:u_data}))
        print(loss_value)
        if loss_value < 0.00001:
            exit()
    if i % 100 == 0:
        if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)
        saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME)) 

    fit_data = np.append(t_data, x_data, axis = 1)
    fit_data = np.append(fit_data, predict, axis = 1)
    # print(fit_data)
    # np.savetxt('fit_data.txt',fit_data,fmt='%f')
    # time.sleep(3)
    # print(t_data)
    # print(x_data)
    # print(u_data)
