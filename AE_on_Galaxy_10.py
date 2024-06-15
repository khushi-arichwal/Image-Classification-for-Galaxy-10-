import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models, losses
from astroNN.models import Galaxy10CNN
from astroNN.datasets import load_galaxy10sdss

# Loading the data
images, labels = load_galaxy10sdss()

# Preprocessing the data
image_dataset = np.empty((len(images), 68, 68, 3))
for i in range(len(image_dataset)):
    image_dataset[i] = images[i, 1:, 1:, :]
    image_dataset[i] = image_dataset[i] / 255.0

image_dataset = np.transpose(image_dataset, (0, 3, 1, 2))

# Splitting the dataset 
split_index = int(0.80 * len(image_dataset))
train_images, test_images = image_dataset[:split_index], image_dataset[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Defining the autoencoder model
class Autoencoder(models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Compiling and training the model
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_images, train_images, epochs=10, shuffle=True, validation_data=(test_images, test_images))

# Displaying a few images and their reconstructions
num_images = 5
plt.figure(figsize=(10, 6))
for i in range(num_images):
    # Display original
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(test_images[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Displaying reconstruction
    ax = plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(autoencoder(test_images[i][np.newaxis])[0])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
