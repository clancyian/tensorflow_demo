# Siraj's tensorflow mnist
# Import MNIST data
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import argparse

parser = argparse.ArgumentParser(description='McParser')

parser.add_argument('-d', '--debug', action='store_true', help='Enables tensorflow and python debugging')
parser.add_argument('-l', '--lr', nargs='?', const=0.01, type=float, default=0.01 , help='Sets the learning rate')
parser.add_argument('-i', '--iterations', nargs='?', const=10, type=int, default=10, help='Sets the number of iterations')
parser.add_argument('-b', '--batch_size', nargs='?', const=100, type=int, default=100, help='Sets the number of iterations')
parser.add_argument('-sd', '--standard_deviation', nargs='?', const=0.0, type=float, default=0.0, help='Sets standard deviation of the initialised weights')
args = parser.parse_args()

# Set parameters
learning_rate = args.lr
training_iteration = args.iterations
batch_size = args.batch_size
std_dev = args.standard_deviation
display_step = round(training_iteration/5)
image_dim = 784
label_dim = 10

graph_name = 'run' + '_' + str(std_dev) + '_' + str(learning_rate) + '_' + str(training_iteration) + '_' + str(batch_size)

# TF graph input
x = tf.placeholder("float", [None, image_dim], name="image") # mnist data image of shape 28*28=image_dim
y = tf.placeholder("float", [None, label_dim], name="label") # 0-9 digits recognition => label_dim classes

# Create a model

# Set model weights
W = tf.Variable(tf.random_normal([image_dim, label_dim],stddev=std_dev), name="W")
b = tf.Variable(tf.random_normal([label_dim],stddev=std_dev), name="b")

#x = tf.random_normal([],stddev=0.01)

with tf.name_scope("Wx_b") as scope:
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Add summary ops to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b)

# More name scopes will clean up graph representation
with tf.name_scope("loss") as scope:
    # Minimize error using cross entropy
    # Cross entropy
    # http://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/
    loss = -tf.reduce_sum(y*tf.log(model))
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
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            # Passing a tensor so returning an ndarray
            avg_cost += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for every label_dim0th iteration
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

