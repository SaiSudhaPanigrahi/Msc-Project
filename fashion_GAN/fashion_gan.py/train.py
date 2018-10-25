import tensorflow as tf
from gan_model import DCGAN

import sys

# write all the output to file
'''original_std = sys.stdout
f  = open ('./output.txt','w')
sys.stdout = f'''

flags  = tf.app.flags
tf.logging.set_verbosity(tf.logging.ERROR)


flags.DEFINE_integer("epoch", 2000, "Epoch to train [250]")
flags.DEFINE_integer("batch_size", 32, "The number of batch images [64]")
flags.DEFINE_integer("sample_size", 28, "The number of sample images [64]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color. [3]")
flags.DEFINE_integer("save_step", 100, "The interval of saving checkpoints. [500]")
flags.DEFINE_string("dataset", "fashion_mnist", "The name of dataset [fashion_mnist, mnist]")
flags.DEFINE_string("job_dir", "checkpoints/gan", "Directory name to save the checkpoints [checkpoints]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_bool("tpu", False, "a flag to set tpu environment set to True if you run this on colab [False]")

FLAGS = flags.FLAGS



def main(_):
    # Print flags
    print("\n---------FLAGS-----------")
    for flag, _ in FLAGS.__flags.items():
        print('"{}": {}'.format(flag, getattr(FLAGS, flag)))
      
    print("--------------------\n")

    dcgan = DCGAN()
    dcgan()
    '''sys.stdout = original_std
    f.close()'''

   

if __name__ == '__main__':
    tf.app.run()