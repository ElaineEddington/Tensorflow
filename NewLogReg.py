from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 100
display_step = 1

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

w = tf.Variable(tf.zeros([784, 10], dtype=tf.float32))
b = tf.Variable(tf.ones([10], dtype=tf.float32))
y_hat = tf.add(b, tf.matmul(x, w))


# Construct model
pred = tf.nn.softmax(tf.matmul(x, w) + b)   # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()  # defaults to saving all variables - in this case w and b

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())


    for epoch in range(training_epochs):
         avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
         # Loop over all batches
         for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                                  y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                saver.save(sess,"/Users/mac/PycharmProjects/untitled1/MyModel",write_meta_graph=True)


    else:
        # Here's where you're restoring the variables w and b.
        # Note that the graph is exactly as it was when the variables were
        # saved in a prior training run.
        ckpt = tf.train.get_checkpoint_state("/Users/mac/PycharmProjects/untitled1")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found")

        # Now you can run the model to get predictions
        batch_x =
        predictions = sess.run(y_hat, feed_dict={x: batch_x})