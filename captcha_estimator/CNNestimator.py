from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import data_utils.base as input_data


Modes = tf.estimator.ModeKeys



def cnn_n_model(features, labels, mode, params):


    LABEL_SIZE = params['label_size']
    NUM_PER_IMAGE = params['num_per_image']
    IMAGE_HEIGHT = params['height']
    IMAGE_WIDTH = params['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))


    input = tf.feature_column.input_layer(features, params['feature_columns'])
    conv1 = tf.layers.conv2d(
                            inputs=input,
                            filters=32,
                            kernel_size=[1, 1],
                            padding='same',
                            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
                            inputs=pool1,
                            filters=64,
                            kernel_size=[3, 3],
                            padding='same',
                            activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1,IMAGE_WIDTH*IMAGE_HEIGHT*4])

    fc1 = tf.layers.dense(inputs=pool2_flat, units=[IMAGE_WIDTH*IMAGE_HEIGHT*4,1024], activation=tf.nn.relu)
    fc1_drop = tf.layers.dropout(inputs=fc1, rate=0.3, training=(mode == Modes.TRAIN))


    logits = tf.layers.dense(inputs=fc1_drop, units= NUM_PER_IMAGE * LABEL_SIZE, activation=None)
    predicted_classes = tf.argmax(logits)

    if mode == Modes.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    if mode in (Modes.TRAIN, Modes.EVAL):
        # Compute loss.
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,predictions=predicted_classes, name='accuracy_op')

    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == Modes.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)







