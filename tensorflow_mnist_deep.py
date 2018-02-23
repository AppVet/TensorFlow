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

"""Deep MNIST for Experts.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
"""


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
# Read MNIST data (split below into mnist.train, mnist.test and mnist.validation)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

""""FLAGS = None"""
import tensorflow as tf
sess = tf.InteractiveSession()

#Create input images and output classes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

#Set weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#Session
sess.run(tf.global_variables_initializer())

#y is the neuron
y = tf.matmul(x,W) + b

#Use cross_entropy
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

######## TRAIN ########

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#Run 1000 epochs
for _ in range(1000):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  
#Evaluate
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
