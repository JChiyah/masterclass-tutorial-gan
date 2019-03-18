"""
    This file contains code to run a Generative Adversarial Network (GAN) using TensorFlow.

    This file is adapted from a DataCamp Community Tutorial
    https://www.datacamp.com/community/tutorials/generative-adversarial-networks
    
    Original author: Stefan Hosein
    Date: May 9th, 2018

    Modified by Javier Chiyah for a Masterclass on Deep Neural Networks
    Heriot-Watt University, 2019
    
    Masterclass date: March 18th, 2019

    You are free to use this file and make changes as required.
    Please, acknowledge the original author above when doing so.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU, ReLU, Softmax
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import initializers


# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"

# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)

# The dimension of our random noise vector.
random_dim = 100


def load_minst_data():
    """
    Loads and returns the MNIST dataset.
    It changes the shape of the images (28x28) 
    to be a vector of 1 dimension (1x784)
    to simplify the neural networks.

    :return: tuple (train data, train data, test data, test data)
    """
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    return x_train, y_train, x_test, y_test


def get_optimiser():
    """
    Gets the optimiser for the neural networks

    :return: keras.optimiser
    """
    return Adam(lr=0.0002, beta_1=0.5)


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    """
    Helper function that plots the generated images into a wall of MNIST images

    :param epoch: number of current epoch
    :param generator: neural network of the Generator
    :param examples: size of the sample
    :param dim: dimensions
    :param figsize: size of the plot
    :return:
    """
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    # generate some fake MNIST images with the current generator network
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    # save the figure with the following name
    plt.savefig('gan_generated_image_epoch_{}.png'.format(epoch))


def train(epochs=1):
    """
    Main function that trains the GAN network for as many epochs as given.
    It will plot generated images every 5 epochs into a file in the same folder.

    :param epochs: number of epochs to train the network for
    """
    batch_size = 128
    # Get the training and testing data
    x_train, y_train, x_test, y_test = load_minst_data()
    # Split the training data into batches of size 128
    batch_count = x_train.shape[0] // batch_size

    # Build our GAN network
    generator = get_generator()
    discriminator = get_discriminator()
    gan = get_gan_network(discriminator, generator)

    # Iterate for as many epochs as necessary
    # At each iteration, the GAN network is trained with a random sample
    # Both the Generator and Discriminator are trained
    for e in range(1, epochs+1):
        print('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in tqdm(range(batch_count)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Generate fake MNIST images
            generated_images = generator.predict(noise)
            all_images = np.concatenate([image_batch, generated_images])

            # Labels for generated and real data
            y_dis = np.zeros(2*batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9

            # Train discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(all_images, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        # every 5 epochs, generate a bunch of random images and plot them
        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)



def get_generator():
    """
    Gets the Generator network of the GAN

    :return: sequential neural network
    """
    generator = Sequential()

    # Layers that may be useful here:
    # Dense() -> fully connected layer
    # More info: https://keras.io/layers/core/

    # Example layer: generator.add(Dense(<size>))

    # Activations functions:
    # Sigmoid
    # Tanh
    # ReLU
    # LeakyReLU

    # Example activation function: generator.add(ReLu())


    # Code here! You need to:

    # 1. create an input layer

    # 2. create hidden layers (2 recommended)

    # 3. create output layer

    # end!

    generator.compile(loss='binary_crossentropy', optimizer=get_optimiser())
    return generator


def get_discriminator():
    """
    Gets the Discriminator network of the GAN

    :return: sequential neural network
    """
    discriminator = Sequential()

    # Layers that may be useful here:
    # Dense() -> fully connected layer
    # Dropout() -> helps to avoid overfitting in classification
    # More info: https://keras.io/layers/core/

    # Example layer: generator.add(Dense(<size>))

    # Activations functions:
    # Sigmoid
    # Tanh
    # ReLU
    # LeakyReLU

    # Example activation function: generator.add(ReLu())


    # Code here! You need to:

    # 1. create an input layer

    # 2. create hidden layers (2 recommended)

    # 3. create output layer

    # end!

    discriminator.compile(loss='binary_crossentropy', optimizer=get_optimiser())
    return discriminator


def get_gan_network(discriminator, generator):
    """
    Sets up the Generative Adversarial Network

    :param discriminator: Discriminator of the GAN
    :param generator: Generator of the GAN
    :return keras.model: GAN network model
    """
    # We initially set trainable to False since we only want to train either the
    # generator or discriminator at a time
    discriminator.trainable = False
    # gan input (noise) will be 100-dimensional vectors
    gan_input = Input(shape=(random_dim,))

    # the output of the generator (an image)
    ## set up generator
    gen_model = # your generator

    # get the output of the discriminator (probability if the image is real or not)
    ## set up discriminator
    disc_model = # your discriminator

    ## create DNN model here
    gan = # your final model

    gan.compile(loss='binary_crossentropy', optimizer=get_optimiser())
    return gan


if __name__ == '__main__':
    # train(number of epochs to train for)
    # each epoch takes around 1 minute
    train(100)

