from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf

COLUMNS = ['target_point', 'heading', 'velocity', 'acceleration', 'stay_point']
LABEL_COLUMN = "label"
CATEGORICAL_COLUMNS = ['stay_point']
CONTINUOUS_COLUMNS = ['heading', 'velocity', 'acceleration']

def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  """Train and evaluate the model."""
#  train_file_name, test_file_name = maybe_download(train_data, test_data)
  df_train = pd.read_csv(
      tf.gfile.Open('./src/data/training_data.csv'),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      sep=';',
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open('./src/data/test_data.csv'),
      names=COLUMNS,
      skipinitialspace=True,
      skiprows=1,
      sep=';',
      engine="python")

  # remove NaN elements
  df_train = df_train.dropna(how='any', axis=0)
  df_test = df_test.dropna(how='any', axis=0)

  df_train[LABEL_COLUMN] = (
      df_train["target_point"].apply(lambda x: "yes" in x)).astype(int)
  df_test[LABEL_COLUMN] = (
      df_test["target_point"].apply(lambda x: "yes" in x)).astype(int)

  print(df_train)
  print(df_test)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=200,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)