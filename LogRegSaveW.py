
from __future__ import print_function

import tensorflow as tf
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
import random
import sys
import numpy as np
from LogReg import accuracy
from LogReg import W
from LogReg import x,y


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def restore(model_file):

    with tf.Session() as sess:

        new_saver = tf.train.import_meta_graph(model_file + ".meta")
        new_saver.restore(sess, model_file)

        with tf.variable_scope("foo", reuse=True):

            temp_var = tf.get_variable("W")
            size_2a = tf.get_variable("b")
            s1 = tf.shape(temp_var).eval()[0]
            s2 = tf.shape(size_2a).eval()[0]

            print("W_old", temp_var.eval())
            ones_mask = tf.ones([s1,s2])
            indices = tf.slice(ones_mask,[0,0],[s1/2,s2/2])

            # turn 'ones_mask' into 1d variable since "scatter_update" supports linear indexing only
            ones_flat = tf.Variable(tf.reshape(ones_mask, [-1]))
            indices_flat = tf.Variable(tf.reshape(indices, [-1]))

            # get linear indices
            linear_indices = tf.random_uniform(tf.shape(indices_flat), dtype=tf.int32, minval=0, maxval =s1*s2-1)
            print("lin_ind",linear_indices)
            # no automatic promotion, so make updates float32 to match ones_mask
            updates = tf.zeros(shape=(tf.shape(linear_indices)), dtype=tf.float32)

            ones_flat_new = tf.scatter_update(ones_flat,linear_indices, updates)

            # convert back into original shape
            ones_mask_new = tf.reshape(ones_flat_new, ones_mask.get_shape())

            W.assign(tf.multiply(W,ones_mask_new))

            print("W_new", W.eval())

            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            new_saver.save(sess, model_file)

            print("Accuracy_new:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

restore('./MyModel2')


