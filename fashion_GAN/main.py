from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape

import tensorflow as tf
import numpy as np

flags  = tf.app.flags
flags.DEFINE_string("label", "trouser", "The name of label to generate the correspond image ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']")
FLAGS = flags.FLAGS
LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']


# Create a wall of generated images
def plot_images(imgs , dim=(5,5), figsize=(5, 5)):
    plt.figure(figsize=figsize)
    for i in range(imgs.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(imgs[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()



def main(_):

    label  = " "   
    runs =0
    fashion_cnn =  Sequential()
    fashion_cnn.add(Reshape((28,28,1)))
    # load the generator model        
    generator  = load_model("../fashion_GAN/weights/dcgan_generator.h5")
    # load the cnn model and add it to Reshape layer - because it trained on GPU
    model = load_model("../fashion_GAN/weights/fashion-cnn-weights.best.hdf5")
    fashion_cnn.add(model)
    #fashion_cnn = keras.models.Model(keras.layers.Reshape((28,28,1)), model.output)
    
    
    while label != FLAGS.label:
        runs +=1
        noise = np.random.normal(0, 1, size=[1, 100]) # generate one image 
        imgs = generator.predict(noise)
        
        score  = fashion_cnn.predict(imgs) 
        indx  = list(score[0]).index(max(score[0]))
        #print(score[0])
        #print(indx)
        label = LABEL_NAMES[indx]
    print("Runs:",runs)  
    print(label)
    plot_images(imgs)  


if __name__ == '__main__':
   tf.app.run()  