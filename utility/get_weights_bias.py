import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time

def ooops():
    print('ooops')

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



# create data
x_data = np.linspace(-1,1,100)[:, np.newaxis] 
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col
ys = tf.placeholder(tf.float32, [None, 1]) # * rows, 1 col

IN_NODE = 1
H1_NODE = 3
H2_NODE = 3
OUT_NODE = 1
LR = 0.1
try:
    temp = np.loadtxt('Weights1_best.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([IN_NODE,H1_NODE])
    print(temp,temp.shape)
    Weights1 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    Weights1 = tf.Variable(tf.random_normal([IN_NODE,H1_N[IN_NODE,H1_NODE]]))

try:
    temp = np.loadtxt('biases1_best.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([1,H1_NODE])
    biases1 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    biases1 = tf.Variable(tf.zeros([1,H1_NODE]) + 0.1)
Wx_plus_b = tf.matmul(xs, Weights1) + biases1
l1 = tf.nn.sigmoid(Wx_plus_b)




try:
    temp = np.loadtxt('Weights2_best.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([H1_NODE,H2_NODE])
    Weights2 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    Weights2 = tf.Variable(tf.random_normal([H1_NODE,H2_NODE]))

try:
    temp = np.loadtxt('biases2_best.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([1,H2_NODE])
    biases2 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    biases2 = tf.Variable(tf.zeros([1,H2_NODE]) + 0.1)
Wx_plus_b = tf.matmul(l1, Weights2) + biases2
l2 = tf.nn.sigmoid(Wx_plus_b)



try:
    temp = np.loadtxt('Weights3_best.txt', dtype=np.float32)
    temp = temp[:,np.newaxis].reshape([H2_NODE,OUT_NODE])
    Weights3 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    Weights3 = tf.Variable(tf.random_normal([H2_NODE,OUT_NODE]))

try:
    temp = np.loadtxt('biases3_best.txt', dtype=np.float32)
    # It's special because temp is number, which is diffenrent 
    # from above
    temp = temp.reshape([1,1])
    biases3 = tf.Variable(temp)
    print('Well Done')
except Exception as e:
    ooops()
    biases3 = tf.Variable(tf.zeros([1,OUT_NODE]) + 0.1)
Wx_plus_b = tf.matmul(l2, Weights3) + biases3
prediction = Wx_plus_b



loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), 
                reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(LR).minimize(loss)

init = tf.global_variables_initializer()



fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()


with tf.Session() as sess:
    sess.run(init)
    for i in range (1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 20 == 0:
            print(sess.run(loss,feed_dict={xs:x_data, ys:y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception :
                pass
            prediction_value = sess.run(prediction, feed_dict={xs:x_data, ys:y_data})
            lines = ax.plot(x_data, prediction_value, 'r-', lw = 4)
            plt.pause(0.1)
    Weights1_best = Weights1.eval()
    biases1_best = biases1.eval()
    Weights2_best = Weights2.eval()
    biases2_best = biases2.eval()
    Weights3_best = Weights3.eval()
    biases3_best = biases3.eval()
    print(Weights1_best)
    print(biases1_best)

    print(Weights2_best)
    print(biases2_best)

    print(Weights3_best)
    print(biases3_best)

    np.savetxt('Weights1_best.txt',Weights1_best,fmt='%f')
    np.savetxt('biases1_best.txt',biases1_best,fmt='%f')
    np.savetxt('Weights2_best.txt',Weights2_best,fmt='%f')
    np.savetxt('biases2_best.txt',biases2_best,fmt='%f')
    np.savetxt('Weights3_best.txt',Weights3_best,fmt='%f')
    np.savetxt('biases3_best.txt',biases3_best,fmt='%f')