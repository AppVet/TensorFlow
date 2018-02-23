'''
Created on Dec 4, 2017

@author: steve
'''

# basics.py
# Some examples from: 
# http://python4java.necaiseweb.org/Fundamentals/UserInteraction
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves.urllib.request import urlopen

import numpy as np
import tensorflow as tf

# Data sets
HOUSE_TRAIN = "C:\\machine-learning-datasets\\housing_test_good\\train.csv"
HOUSE_TRAIN_URL = "http://download.tensorflow.org/data/HOUSE_TRAIN.csv"

HOUSE_TEST = "C:\\machine-learning-datasets\\housing_test_good\\test.csv"
HOUSE_TEST_URL = "http://download.tensorflow.org/data/HOUSE_TEST.csv"

def main():
   # If the training and test sets aren't stored locally, download them.
   if not os.path.exists(HOUSE_TRAIN):
     print("Could not find training file: ", HOUSE_TRAIN)
     raw = urlopen(HOUSE_TRAIN_URL).read()
     with open(HOUSE_TRAIN, "wb") as f:
        f.write(raw)
   else:
     print("Found training file: ", HOUSE_TRAIN)
   if not os.path.exists(HOUSE_TEST):
      print("Could not find training file: ", HOUSE_TEST)
      raw = urlopen(HOUSE_TEST_URL).read()
      with open(HOUSE_TEST, "wb") as f:
         f.write(raw)
   else:
     print("Found test file: ", HOUSE_TRAIN)
     
   print(HOUSE_TRAIN)
   
  # Load datasets.
   training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=HOUSE_TRAIN,
      target_dtype=np.int,
      features_dtype=np.float32)
   test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=HOUSE_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
   feature_columns = [tf.feature_column.numeric_column("x", shape=[4])]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
   classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                          hidden_units=[10, 20, 10],
                                          n_classes=3,
                                          model_dir="/tmp/iris_model")
  # Define the training inputs
   train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train model.
   classifier.train(input_fn=train_input_fn, steps=2000)

  # Define the test inputs
   test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  # Evaluate accuracy.
   accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]

   print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
   new_samples = np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)
   predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": new_samples},
      num_epochs=1,
      shuffle=False)

   predictions = list(classifier.predict(input_fn=predict_input_fn))
   predicted_classes = [p["classes"] for p in predictions]

   print(
      "New Samples, Class Predictions:    {}\n"
      .format(predicted_classes))

if __name__ == "__main__":
   print("Running 'CSV test.py'")
   main()