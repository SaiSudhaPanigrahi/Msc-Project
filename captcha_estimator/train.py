import tensorflow as tf
import argparse
import data_utils.base as INPUT
import captcha_estimator.CNNestimator as cnn
import sys



global FLAGS


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()



def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    inputs = features if labels is None else (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the dataset.
    return dataset



def main(argv):
    global FLAGS
    # load data
    # Todo: Change DATE_DIR to FLAG choice'
    meta, train, test = INPUT.load_data(FLAGS.data_dir, flatten=False)
    print('data loaded')
    print('train images: %s. test images: %s' % (train[0].shape[0], test[0].shape[0]))

    # --------------------------------------------
    #  configure setup variables
    # -------------------------------------------

    LABEL_SIZE = meta['label_size']
    NUM_PER_IMAGE = meta['num_per_image']
    IMAGE_HEIGHT = meta['height']
    IMAGE_WIDTH = meta['width']
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))

    feature_column = [tf.feature_column.numeric_column(key="image",
                                                       shape=IMAGE_WIDTH*IMAGE_HEIGHT)]

 # the main part - define our custom classifier
    classifier = tf.estimator.Estimator(
            model_fn=cnn.cnn_n_model,
            params={
                'feature_columns':feature_column,
                'label_size': LABEL_SIZE,
                'num_per_image':NUM_PER_IMAGE,
                'height':IMAGE_HEIGHT,
                'width':IMAGE_WIDTH
                },
            model_dir='log/cnn'
        )

    classifier.train(
        input_fn=lambda: train_input_fn(train[0], train[1], FLAGS.batch_size),
        steps=FLAGS.train_steps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir', type=str, default='../images/2X5/',help='Directory for storing input data')
    parser.add_argument('-bs','--batch_size', default=100, type=int, help='batch size')
    parser.add_argument('-ts','--train_steps', default=1000, type=int,help='number of training steps')
    FLAGS =  parser.parse_args()
    tf.app.run(main=main)


