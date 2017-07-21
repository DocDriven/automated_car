"""First Implementation of an Neural Network for Predicting Pedestrians"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import pandas as pd
import tensorflow as tf


# Define path to data sets
PATH = './src/data/'
TRAINING_FILE_NAME = 'training_data_pedestrian.csv'
TEST_FILE_NAME = 'test_data_pedestrian.csv'


COLUMNS = ['target_point', 'heading', 'velocity', 'acceleration', 'stay_point']
LABEL_COLUMN = 'label'
CATEGORICAL_COLUMNS = ['stay_point']
CONTINUOUS_COLUMNS = ['heading', 'velocity', 'acceleration']


def build_estimator(optimizer=None, activation_fn=tf.nn.relu):
    """Build an estimator"""
    # Sparse base columns
    column_stay_point = tf.contrib.layers.sparse_column_with_keys(
            column_name='stay_point',
            keys=['no', 'yes'])
        
    # Continuous base columns
    column_heading = tf.contrib.layers.real_valued_column('heading')
    column_velocity = tf.contrib.layers.real_valued_column('velocity')
    column_acceleration = tf.contrib.layers.real_valued_column('acceleration')
    
    pedestrian_feature_columns = [column_heading, 
                                  column_velocity, 
                                  column_acceleration,
                                  tf.contrib.layers.embedding_column(
                                          column_stay_point, 
                                          dimension=8, 
                                          initializer=tf.truncated_normal_initializer)]
        
    # Create classifier (m)
    estimator = tf.contrib.learn.DNNClassifier(
            hidden_units=[10],
            feature_columns=pedestrian_feature_columns,
            model_dir='./tmp/pedestrian_model',
            n_classes=2,
            optimizer=optimizer,
            activation_fn=activation_fn)
    
    return estimator


def input_fn(df):
    """Provides an input function"""
    # Create dict mapping from each continous feature columnname (k) to the
    # values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(
            df[k].values, shape=[df[k].size, 1]) for k in CONTINUOUS_COLUMNS}
    
    # Create dict mapping from each categorical feature column name (k) to the
    # values of that column stored in a tf.SparseTensor
    categorical_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}

    # Merge the two dicts into one
    feature_cols = continuous_cols.copy()
    feature_cols.update(categorical_cols)
    
    # Convert label column into constant Tensor
    label = tf.constant(df[LABEL_COLUMN].values)
    
    # Return feature cols and label
    return feature_cols, label


def train_and_eval(model_dir, train_steps):
    """Train and evaluate the model"""
    # Create data frame objects for training and test data sets
    df_train = pd.read_csv(tf.gfile.Open(PATH+TRAINING_FILE_NAME),
                           names=COLUMNS,
                           skipinitialspace=True,
                           skiprows=1,
                           sep=';',
                           engine="python")
    
    df_test = pd.read_csv(tf.gfile.Open(PATH + TEST_FILE_NAME), 
                          names=COLUMNS, 
                          skipinitialspace=True, 
                          skiprows=1,
                          sep=';',
                          engine='python')

    # Remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    
    # Decode strings binary
    df_train[LABEL_COLUMN] = (
            df_train['target_point'].apply(lambda x: 'yes' in x)).astype(bool)
     
    df_test[LABEL_COLUMN] = (
            df_test['target_point'].apply(lambda x: 'yes' in x)).astype(bool)

    print(df_train.dtypes)
#    optimizer = tf.train.
    
    estimator = build_estimator(activation_fn=tf.sigmoid)
    estimator.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = estimator.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
#    results = estimator.evaluate(input_fn=lambda: input_fn(df_test))
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

def main():
    train_and_eval('./tmp/pedestrian_model/', 200)

if __name__ == "__main__":
    main()

#FLAGS = None

#def main(_):
#    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
#                 FLAGS.train_data, FLAGS.test_data)
#
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.register("type", "bool", lambda v: v.lower() == "true")
#    parser.add_argument(
#            "--model_dir",
#            type=str,
#            default="",
#            help="Base directory for output models."
#            )
#    parser.add_argument(
#            "--model_type",
#            type=str,
#            default="wide_n_deep",
#            help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
#            )
#    parser.add_argument(
#            "--train_steps",
#            type=int,
#            default=200,
#            help="Number of training steps."
#            )
#    parser.add_argument(
#            "--train_data",
#            type=str,
#            default="",
#            help="Path to the training data."
#            )
#    parser.add_argument(
#            "--test_data",
#            type=str,
#            default="",
#            help="Path to the test data."
#            )
#    FLAGS, unparsed = parser.parse_known_args()
#    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)​
