# coding: utf-8

import tensorflow as tf
import time


H = 30
BATCH_SIZE = 100
ITERATION = 10000
LAMBDA = 0.00001
BETA = 3.0
RHO = 0.05
LEARNING_RATE = 0.1


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Input: x : 41
a = [1000, 0, 1, 2, 9, 6, 0, 0, 0, 2,
     31, 3, 9, 1, 2, 1, 0, 2, 0, 0,
     4, 0, 0, 0, 0.2, 2, 1, 0, 0, 3,
     6, 2, 10, 2, 0, 0, 450, 0, 1, 2,
     0.1]
x = tf.placeholder(tf.float32, [None, 41])

# Ground Truth y_ : 41
b = [200, 0, 0, 0, 9, 6, 0, 0, 1, 50,
     1, 3, 2, 1, 2, 1, 0, 2, 0, 0,
     0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
     2, 1, 50, 2, 0, 0, 0, 0, 9, 20,
     1]
y_ = tf.placeholder(tf.float32, [None, 41])

# Variable: W, b1
W1 = weight_variable((41, H))
b1 = bias_variable([H])

# Hidden Layer: h
# softsign(x) = x / (abs(x)+1);
h = tf.nn.softsign(tf.matmul(x, W1) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)

# Variable: b2
W2 = tf.transpose(W1)  # 転置
b2 = bias_variable([41])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# Define Loss Function
rho_cap = tf.reduce_sum(h) / H
kl_div = BETA * tf.reduce_sum(RHO * tf.log(RHO / rho_cap) + (1 - RHO) * tf.log((1 - RHO) / (1 - rho_cap)))
weight_sum = tf.reduce_sum(tf.pow(W1, 2)) + tf.reduce_sum(tf.pow(W2, 2))
bias_sum = tf.reduce_sum(tf.pow(b1, 2)) + tf.reduce_sum(tf.pow(b2, 2))
weight_bias_decay = LAMBDA * 0.5 * (weight_sum + bias_sum)
weight_decay = LAMBDA * 0.5 * weight_sum
l2 = tf.nn.l2_loss(y - y_) / BATCH_SIZE
loss = l2

# Use Adan Optimizer
# train_step = tf.train.AdamOptimizer().minimize(loss)
train_step = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training
start = time.time()
for step in range(ITERATION):
    batch_xs, batch_ys = a, a
    sess.run(train_step, feed_dict={x: [batch_xs], y_: [batch_ys], keep_prob: 1.0})

    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: [batch_xs], y_: [batch_ys], keep_prob: 1.0}))

print("Learning Time: " + str(time.time() - start) + "[s]\n")

print("Input:\n" + str(a))
print("Ground Truth:\n" + str(b))
p = y.eval(session=sess, feed_dict={x: [a], y_: [a],  keep_prob: 1.0})

print("Learning Result:\n" + str(p))
print("Difference:\n" + str(p - a))
