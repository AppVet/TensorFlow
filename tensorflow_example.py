from __future__ import print_function
'''
Created on Dec 4, 2017

@author: steve
@see: http://python4java.necaiseweb.org/Collections/AdvancedListFeatures
'''
# basics.py
# Some examples from: 
# http://python4java.necaiseweb.org/Collections/AdvancedListFeatures
if __name__ == '__main__':
   print("Running 'tensorflow1.py'")

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

#Example: Create two nodes
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 
print(node1, node2)

#Run TF session to view values
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))