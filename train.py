# Siraj's tensorflow mnist
# Import MNIST data
import math
import argparse
from models import lenet, simple
import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
from tensorflow.python import debug as tf_debug

config = tf.ConfigProto()
# Ensure only needed memory is allocated by the GPU
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser(description='McParser')

parser.add_argument('-d', '--debug', action='store_true', help='Enables tensorflow and python debugging')
parser.add_argument('-l', '--lr', nargs='?', const=0.01, type=float, default=0.01, help='Sets the learning rate')
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
display_step = math.ceil(training_iteration/10)

graph_name = 'run' + '_' + str(args.model) + '_' + str(std_dev) + '_' + str(learning_rate) \
    + '_' + str(training_iteration) + '_' + str(batch_size) + '_' + str(args.dropout)

# TF graph input
x = tf.placeholder(tf.float32, [None, 784], name="image")  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10], name="label")  # 0-9 digits recognition => label_dim classes
keep_prob = tf.placeholder(tf.float32, name="dropout")

# Create a model

with tf.name_scope("Model") as scope:
    if args.model == 1:
        # Lenet
        model = lenet(x, args.dropout)
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

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Test the model
predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(predictions, "float"))

with tf.name_scope("validation") as scope:
    # Add a scalar to record accuracy
    tf.summary.scalar("accuracy", accuracy)

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()


# Launch the graph
with tf.Session(config=config) as sess:
    if args.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # Q why is b not initialised?

    sess.run(init)

    summary_writer = tf.summary.FileWriter('./graph/' + graph_name, sess.graph)

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
            # if i % 100:
            #    summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            #    summary_writer.add_summary(summary_str, iteration*total_batch + i)

        # Get a random batch of data from the training data set
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        summary_train = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
        summary_val = sess.run(merged_summary_op, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        # acc = merged_summary_op.eval({x: mnist.validation.images, y: mnist.validation.labels})
        summary_writer.add_summary(summary_val, iteration + 1)
        summary_writer.add_summary(summary_train, iteration + 1)
        # Display logs per iteration step
        if (iteration + 1) % display_step == 0:
            print("Iteration:", '%02d' % (iteration + 1), "Average cost:", "{:.9f}".format(avg_cost),
                  "Accuracy Training:"
                  "{:1.4f}".format(accuracy.eval({x: batch_xs, y: batch_ys})),
                  "Accuracy Validation:"
                  "{:1.4f}".format(accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels})))

    print("Training completed!")
    save_file = "./weights/" + graph_name
    save_path = saver.save(sess, save_file)
    print("Model saved in file: %s" % save_path)

    if args.debug:
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    print("Test Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
