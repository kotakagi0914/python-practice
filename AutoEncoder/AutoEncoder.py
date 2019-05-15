# coding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
H = 50
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Input: x : 28*28=784
x = tf.placeholder(tf.float32, [None, 784])

# Variable: W, b1
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1);
h = tf.nn.softsign(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([784])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# Define Loss Function
loss = tf.nn.l2_loss(y-x) / BATCH_SIZE

# For tensorboard learning monitoring
tf.scalar_summary("l2_loss", loss)

# Use Adan Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('summary/l2_loss', graph_def=sess.graph_def)

# Training
for step in range(2000):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x: batch_xs, keep_prob: (1 - DROP_OUT_RATE)})
    # a = sess.run(x, feed_dict={x: batch_xs})
    # Collect Summary
    summary_op = tf.merge_all_summaries()
    summary_str = sess.run(summary_op, feed_dict={x: batch_xs, keep_prob: 1.0})
    summary_writer.add_summary(summary_str, step)
    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))

# Draw Encode/Decode Result
N_COL = 10
N_ROW = 2
plt.figure(figsize=(N_COL, N_ROW*2.5))
batch_xs, _ = mnist.train.next_batch(N_COL*N_ROW)
for row in range(N_ROW):
    for col in range(N_COL):
        i = row*N_COL + col
        data = batch_xs[i:i+1]

        # Draw Input Data(x)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL+col+1)
        plt.title('IN:%02d' % i)
        plt.imshow(data.reshape((28, 28)), cmap="magma", clim=(0,1.0), origin='upper')
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

        # Draw Output Data(y)
        plt.subplot(2*N_ROW, N_COL, 2*row*N_COL + N_COL+col+1)
        plt.title('OUT:%02d' % i)
        y_value = y.eval(session=sess, feed_dict={x: data, keep_prob: 1.0})
        plt.imshow(y_value.reshape((28, 28)), cmap="magma", clim=(0, 1.0), origin='upper')
        plt.tick_params(labelbottom="off")
        plt.tick_params(labelleft="off")

plt.savefig("result.png")
plt.show()
