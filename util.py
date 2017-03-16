import tensorflow as tf

def build_model(x, y, training=True, reuse=None, num_filters=16, ksize=4, stride=2):
    with tf.variable_scope('model'):
        x = tf.reshape(x, [-1, 28, 28, 1])
        x = tf.layers.conv2d(x, num_filters, ksize, stride,
                            use_bias=False, padding='same',
                            reuse=reuse, name='conv1')
        x = tf.layers.batch_normalization(x,
                            scale=False, training=training,
                            reuse=reuse, name='bn1')
        x = tf.nn.relu(x, name='relu1')

        x = tf.layers.conv2d(x, num_filters, ksize, stride, padding='same',
                            reuse=reuse, name='conv2')

        logits = tf.layers.conv2d(x, 10, x.shape[1:3].as_list(), reuse=reuse, name='logits')
        y = tf.reshape(y, [-1, 1, 1, 10], name='labels')

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))

        predictions = tf.argmax(logits, axis=-1, name='predictions')
        correct_predictions = tf.equal(predictions, tf.argmax(y, -1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        return accuracy, loss
