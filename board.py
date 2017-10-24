# Siraj's tensorflow mnist
# Import MNIST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import argparse

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
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

    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return  y_conv

def simple(x):
    # Set model weights
    W = weight_variable([784,10])
    b = bias_variable([10])
    # Add summary ops to collect data
    w_h = tf.summary.histogram("weights", W)
    b_h = tf.summary.histogram("biases", b)

    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
    return model

parser = argparse.ArgumentParser(description='McParser')

parser.add_argument('-d', '--debug', action='store_true', help='Enables tensorflow and python debugging')
parser.add_argument('-l', '--lr', nargs='?', const=0.01, type=float, default=0.01 , help='Sets the learning rate')
parser.add_argument('-i', '--iterations', nargs='?', const=10, type=int, default=10, help='Sets the number of iterations')
parser.add_argument('-b', '--batch_size', nargs='?', const=100, type=int, default=100, help='Sets the number of iterations')
parser.add_argument('-sd', '--standard_deviation', nargs='?', const=0.0, type=float, default=0.0, help='Sets standard deviation of the initialised weights')
parser.add_argument('-m', '--model', nargs='?', const=0, type=int, default=0, help='Sets model')
parser.add_argument('-dr', '--dropout', nargs='?', const=1.0, type=float, default=1.0, help='Sets the dropout rate')
args = parser.parse_args()

# Set parameters
learning_rate = args.lr
training_iteration = args.iterations
batch_size = args.batch_size
std_dev = args.standard_deviation
display_step = round(training_iteration/5)

graph_name = 'run' + '_' + str(args.model) + '_' + str(std_dev) + '_' + str(learning_rate) \
    + '_' + str(training_iteration) + '_' + str(batch_size) + '_' + str(args.dropout)

# TF graph input
x = tf.placeholder(tf.float32, [None, 784], name="image") # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10], name="label") # 0-9 digits recognition => label_dim classes
keep_prob = tf.placeholder(tf.float32, name="dropout")

# Create a model

with tf.name_scope("Model") as scope:
    if args.model == 1:
        # Lenet
        model = lenet(x,args.dropout)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=model))
    else:
        model = simple(x)

        loss = -tf.reduce_sum(y*tf.log(model))

# More name scopes will clean up graph representation
with tf.name_scope("loss") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    # http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
    # or cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    # Create a summary to monitor the cost function
    tf.summary.scalar("summary_loss", loss)

with tf.name_scope("train") as scope:
    # Gradient descent
    # The minimize method computes and applies the gradients
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Q why is b not initialised?

    sess.run(init)

    summary_writer = tf.summary.FileWriter('./graph/' + graph_name,sess.graph)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            # Passing an operation here so no return value
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: args.dropout})
            # Compute the average loss
            # Passing a tensor so returning an ndarray
            avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for every 100th iteration
            if i % 100:
                summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
                summary_writer.add_summary(summary_str, iteration*total_batch + i)

        # Display logs per iteration step
        if (iteration + 1) % display_step == 0:
            print ("Iteration:", '%04d' % (iteration + 1), "Average cost=", "{:.9f}".format(avg_cost))

    print ("Tuning completed!")
    if args.debug:
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

