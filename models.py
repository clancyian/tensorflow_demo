# A couple of simple tensorflow models
import tensorflow as tf

def _weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def _bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def lenet(x,keep_prob):

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Reshape the mnist sequence back to image
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    W_conv1 = _weight_variable([5, 5, 1, 32])
    b_conv1 = _bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = _weight_variable([5, 5, 32, 64])
    b_conv2 = _bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = _weight_variable([7 * 7 * 64, 1024])
    b_fc1 = _bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = _weight_variable([1024, 10])
    b_fc2 = _bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return  y_conv

def simple(x):
    # Set model weights
    W = _weight_variable([784,10])
    b = _bias_variable([10])
    # Add summary ops to collect data
    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    return model


