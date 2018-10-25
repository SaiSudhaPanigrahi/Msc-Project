from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

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



def main():

    generator  = load_model('./fashion/weights/dcgan_generator(1).h5')
    #fashion_cnn = load_model('./weights/fashion-cnn-weights.best.hdf5')


    noise = np.random.normal(0, 1, size=[1, 100]) # generate one image 
    imgs = generator.predict(noise)
   
    plot_images(imgs)
    # change the dim to - (28,28,1)
    imgs =  np.moveaxis(imgs, 0, -1)
    #np.reshape(imgs , (28,28,1))
    #imgs = np.expand_dims(imgs ,0)
    print(imgs.shape)
    #plt.show(imgs[1:])
    #plot_generated_images(imgs)

    '''score  = fashion_cnn.predict(imgs)
    label  = list(score[0]).index(max(score[0]))
    print(score[0])
    print(label)
    print(LABEL_NAMES[label])'''


if __name__ == '__main__':
   main()   