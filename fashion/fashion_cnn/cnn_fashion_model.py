"""
this is an model that only use to detact fashion_mnist images
using tensorflow and kears
"""

import tensorflow as tf
import matplotlib.pyplot as plt 

from tensorflow.keras.layers import (MaxPool2D , Conv2D , 
                                     Dropout , Flatten , 
                                     Dense)
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist 
from tensorflow.keras.callbacks import TensorBoard , ModelCheckpoint

# define tyhe train test and eval data
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()
tensorboard = TensorBoard(log_dir='./fashion_cnn/logs',
                          batch_size=64,
                            write_graph=True,
                            histogram_freq=5,
                            write_images=True,
                            write_grads=True)
checkpointer = ModelCheckpoint(filepath='./fashion_cnn/weights.best.hdf5', verbose = 1, save_best_only=True)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
# data Normalization
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# create a validation set fron the train data
val_size  = int(len(x_train)*0.1)
(x_valid,y_valid) = x_train[:val_size] , y_train[:val_size]

# reshape the data to fit the model
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_valid = x_valid.reshape(x_valid.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
#-------------------------------
# define the model and layers
#------------------------------
model  = Sequential()
model.add(Conv2D(filters=64 , kernel_size =3 , 
                 padding="same" , activation="relu",
                 input_shape = (28,28,1)))
model.add(MaxPool2D(pool_size=3))
model.add(Dropout(0.3))                 
model.add(Conv2D(filters=32 , kernel_size =3 , strides=2,
                 padding="same" , activation="relu"))
model.add(MaxPool2D(pool_size=3))                 
model.add(Flatten())
model.add(Dense(256,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(10,activation="softmax"))

model.summary()


#-------------------------------
# compile the model 
#-----------------------------
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

#-------------------------------
# fit (train) the model 
#-----------------------------
model.fit(x_train, y_train,
            batch_size=64,
            epochs=15,
            validation_data=(x_valid,y_valid),
            #validation_steps=1,
            callbacks=[tensorboard,checkpointer],
            shuffle=True)#,verbose=2)   

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)
# Print test accuracy
print('\n', 'Test accuracy:', score[1])  
if score >=0.9:
    model.save("./fashion_cnn/weights/fashion_%2f.hdf5" % score[1]) 
    print("model saved in 'weights' folder...")                  