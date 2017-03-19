import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import *

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

accu_train, loss_train = build_model(x, y)
accu_val, loss_val = build_model(x, y, reuse=True, training=False)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_train)

batch_size = 64
num_iter = 10000
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for k in range(num_iter):
        if k % (num_iter // 100) == 0:
            print('[%6d/%6d]' % (k, num_iter))
        img, lbl = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: img, y: lbl})

    print('accu_train = %.4f'
            % sess.run(accu_train, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print('accu_val = %.4f'
            % sess.run(accu_val, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
