# coding: utf-8

import tensorflow as tf
import time


H = 10
BATCH_SIZE = 100
ITERATION = 10000


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Input: x : 41
a = ([1000, 0, 1, 2, 9, 6, 0, 0, 0, 2,
     31, 3, 9, 1, 2, 1, 0, 2, 0, 0,
     4, 0, 0, 0, 0.2, 2, 1, 0, 0, 3,
     6, 2, 10, 2, 0, 0, 450, 0, 1, 2,
     0.1],
     [2000, 0.3, 1.2, 0, 0, 6, 0, 0, 0, 50000,
     0, 3, 2, 1, 2, 100, 0, 2, 0, 0.8,
     0, 0, 0, 0, 0.2, 2, 7, 0, 10, 3,
     6, 2.1, 0, 0, 0, 0, 70, 0, 1, 9,
     0.4]
     )

x = tf.placeholder(tf.float32, [None, 41])

# Ground Truth y_ : 41
b = ([200, 0, 0, 0, 9, 6, 0, 0, 1, 50,
     1, 3, 2, 1, 2, 1, 0, 2, 0, 0,
     0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
     2, 1, 50, 2, 0, 0, 0, 0, 9, 20,
     1])

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
loss = tf.nn.l2_loss(y - x) / BATCH_SIZE

# Use Adan Optimizer
train_step = tf.train.AdamOptimizer().minimize(loss)

# Prepare Session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training
start = time.time()
for step in range(ITERATION):
    batch_xs = a
    for i in range(len(a)):
        sess.run(train_step, feed_dict={x: [batch_xs[i]], keep_prob: 1.0})

    # Print Progress
    if step % 100 == 0:
        print(loss.eval(session=sess, feed_dict={x: batch_xs, keep_prob: 1.0}))

print("Learning Time: " + str(time.time() - start) + "[s]\n")
p = y.eval(session=sess, feed_dict={x: a,  keep_prob: 1.0})

print("Input:\n" + str(a))
# print("Ground Truth:\n" + str(b))
print("Learning Result:\n" + str(p))
print("Difference:\n" + str(p - b))
