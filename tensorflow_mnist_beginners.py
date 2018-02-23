# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST Softmax Regression classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None


def main(_):
   
   ####### MODEL SET UP ########
     
  # Read MNIST data (split below into mnist.train, mnist.test and mnist.validation)
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  #x is a placeholder for the number of 28x28=784 pixel image arrays. Here [None,784] is a shape.
  x = tf.placeholder(tf.float32, [None, 784])
  #W is the zeroed-out shape for weights (produce 10-dimensional vectors of evidence)
  W = tf.Variable(tf.zeros([784, 10]))
  #b is the zeroed-out shape for biases 
  b = tf.Variable(tf.zeros([10]))
  #y is the actual model defined in a single line. Here, matrix multiplication xW + b.
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])
  
  ####### TRAINING ########

  # Use 'cross-entropy' function to tell us the error margin of how well our model is training 
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  # Apply gradient descent to reduce loss
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  # Launch the model using a session
  sess = tf.InteractiveSession()
  # Initialize all variables created
  tf.global_variables_initializer().run()
  
  # Train 1000 times (epochs)
  for _ in range(1000):
     #xs is batch of 100 images, ys is batch of 100 labels?
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #Run epoch on the batched images and labels
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print("FINAL ACCURACY: ", sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)