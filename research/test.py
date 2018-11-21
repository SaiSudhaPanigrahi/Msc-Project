from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model, Sequential , load_model
from PIL import Image

import tensorflowjs as tfjs
import numpy as np



LABEL_NAMES = ['t_shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots']

class GAN_CNN(Model):
    def __init__(self):
        super().__init__(name="gan_cnn")
        self.gan = load_model("fashion_GAN/weights/dcgan_generator.h5")
        self.cnn  = load_model("fashion_GAN/weights/fashion-cnn-weights.best.hdf5")
        
        self.model = Sequential([self.gan ,Reshape((28,28,1) ,name="cpu-reshpe") , self.cnn])
        '''for layer in  self.gan.layers:
            super().add(layer)
        super().add(Reshape((28,28,1) ,name="cpu-reshpe") )
        for layer in  self.cnn.layers:
            super().add(layer)'''

        self.imgs = None
        
       
    @property
    def predict(self):
            noise = np.random.normal(0, 1, size=[1, 100]) # generate one image 
            self.imgs = self.gan.predict(noise)
            score = self.model.predict(noise)
            #print(score[0])
            return score   
    
    def save(self, path):
        self.model.save(path)

    def generate_with_label(self,label):
        
        score , runs = self.predict  ,0
        indx  = list(score[0]).index(max(score[0]))
        gen_label = LABEL_NAMES[indx]
        while(gen_label!=label):
            score  = self.predict
            indx  = list(score[0]).index(max(score[0]))
            gen_label = LABEL_NAMES[indx]
            runs += 1

        print(f"the item prediction: {label} | Runs: {runs}"  )
        return label

    def generete_and_imgs(self):
        #if(self.imgs.isEmpty()):
        #    self.predict
        img = Image.fromarray(self.imgs,"RGB")
        img.save("genrated.png")
        img.show()    
        return self.imgs[0]        


model = GAN_CNN()
#tfjs.converters.save_keras_model(model, "tfjs/gan-cnn/")
model.save("gan-cnn.weights.h5" )
model.generate_with_label('dress') 
#model.generete_and_imgs()