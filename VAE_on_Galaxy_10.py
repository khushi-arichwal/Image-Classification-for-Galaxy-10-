# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_probability
from astroNN.models import Galaxy10CNN
from astroNN.datasets import load_galaxy10sdss
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from astroNN.datasets.galaxy10sdss import galaxy10cls_lookup

# Loading the data
images, labels = load_galaxy10sdss()

# Preprocessing the data
image_dataset = np.empty((len(images), 68, 68, 3))
for i in range(len(image_dataset)):
    image_dataset[i] = images[i, 1:, 1:, :]
    image_dataset[i] = image_dataset[i] / 255.0

image_dataset = np.transpose(image_dataset, (0, 3, 1, 2))

# Splitting the data
split_index = int(0.80 * len(images))
train_images, test_images = image_dataset[:split_index], image_dataset[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

# Defining the VAE model
class VariationalAutoencoder(models.Model):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = models.Sequential([
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
        ])
        self.mean_layer = layers.Dense(256)
        self.logvar_layer = layers.Dense(256)
        self.decoder = models.Sequential([
            layers.Dense(512, activation='relu'),
            layers.Dense(17*17*64, activation='relu'),
            layers.Reshape((17, 17, 64)),
            layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation='relu'),
            layers.Conv2DTranspose(16, (4, 4), strides=(2, 2), padding='same', activation='relu'),
            layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid'),
        ])

    def encode(self, input_images):
        encoded_images = self.encoder(input_images)
        mean_encoding = self.mean_layer(encoded_images)
        logvar_encoding = self.logvar_layer(encoded_images)
        return mean_encoding, logvar_encoding

    def decode(self, latent_space):
        return self.decoder(latent_space)

    def reparameterize(self, mean_encoding, logvar_encoding):
        epsilon = tf.random.normal(shape=mean_encoding.shape)
        z = mean_encoding + tf.exp(0.5 * logvar_encoding) * epsilon
        return z

    def call(self, input_images):
        mean_encoding, logvar_encoding = self.encode(input_images)
        z = self.reparameterize(mean_encoding, logvar_encoding)
        return self.decode(z), mean_encoding, logvar_encoding

# Defining the loss function
def variational_autoencoder_loss(original_images, reconstructed_images, mean_encoding, logvar_encoding):
    reconstruction_loss = losses.binary_crossentropy(original_images, reconstructed_images)
    kl_divergence_loss = - 0.5 * tf.reduce_mean(1 + logvar_encoding - tf.square(mean_encoding) - tf.exp(logvar_encoding))
    return reconstruction_loss, kl_divergence_loss

# Compiling and training the model
vae = VariationalAutoencoder()
optimizer = optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=variational_autoencoder_loss)
vae.fit(train_images, train_images, epochs=20, batch_size=64)

# Testing the model
reconstructed_images, mean_encoding, logvar_encoding = vae(test_images)
reconstruction_loss, kl_divergence_loss = variational_autoencoder_loss(test_images, reconstructed_images, mean_encoding, logvar_encoding)
print('Reconstruction loss:', tf.reduce_mean(reconstruction_loss).numpy())
print('KL divergence:', tf.reduce_mean(kl_divergence_loss).numpy())

