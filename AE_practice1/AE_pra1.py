# coding: utf-8

import tensorflow as tf
import numpy as np

H = 7
BATCH_SIZE = 1
DROP_OUT_RATE = 0.0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# input
in_a = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0]])
# Grand Truth
gt = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0]])


# Input: x : 3*3 = 9
x = tf.placeholder(tf.float32, [None, 9])
y_ = tf.placeholder(tf.float32, [None, 9])


# Variable: W, b1
W = weight_variable((9, H))
b1 = bias_variable([H])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1);
h = tf.nn.softsign(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W)  # 転置
b2 = bias_variable([9])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# Define Loss Function
loss = tf.nn.l2_loss(y - y_) / BATCH_SIZE

# For tensorboard learning monitoring
tf.scalar_summary("l2_loss", loss)

# Use Adan Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training
for step in range(1000):
    batch_xs, batch_ys = in_a, gt
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: (1 - DROP_OUT_RATE)})

    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}))

a = y.eval(session=sess, feed_dict={x: in_a, keep_prob: 1.0})
print(a)
