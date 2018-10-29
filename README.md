# M-sc-Project
Deep neural network project from M.sc in software engineering.  
A different approach to text to image synthesis that show in this paper [here]("https://arxiv.org/abs/1605.05396")  

we define a simple approach that trained on fasion_mnist dataset.  
### Architecture: 
 
A simple GAN model .  
A simple CNN

![architecture](./gan.jpg)

* to train the cnn model:  
   run the `fashion_cnn_tpu.ipynb` notebook on colab 

* to train the gan model (V2 is reccomended):  
   run the `fashion_mnist_gan_v1/v2.ipynb` notebook on colab 

* to see the CNN model graph on tensorboard:   
run  `tensorboard --logdir ./fashion_GAN/fashion_cnn/logs` 

* to see the GAN model graph on tensorboard:   
run  `tensorboard --logdir ./fashion_GAN/fashion_gan/logs` 

