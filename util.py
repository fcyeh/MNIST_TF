import tensorflow as tf

def conv_bn_relu(x, num_filters, ksize=3, stride=2, training=True, reuse=None, name='conv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                            use_bias=False, padding='same',
                            reuse=reuse, name='conv2d')
        x = tf.layers.batch_normalization(x,
                            scale=False, training=training,
                            reuse=reuse, name='bn')
        return tf.nn.relu(x, name='relu')


def build_model(x, y, training=True, reuse=None):
    with tf.variable_scope('model'):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = conv_bn_relu(x, 16, reuse=reuse, training=training, name='conv1')
        x = conv_bn_relu(x, 16, reuse=reuse, training=training, name='conv2')

        logits = tf.layers.conv2d(x, 10, x.shape[1:3].as_list(),
                            reuse=reuse, name='logits')
        y = tf.reshape(y, [-1, 1, 1, 10], name='labels')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

        predictions = tf.argmax(logits, axis=-1, name='predictions')
        correct_predictions = tf.equal(predictions, tf.argmax(y, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        return accuracy, loss
