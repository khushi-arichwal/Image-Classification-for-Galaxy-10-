#importing necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import numpy as np
import matplotlib.pyplot as plt
import time

# Loading the dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalizing pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the VAE model
class VariationalAutoEncoder(models.Model):
    def __init__(self, latent_dimension=192):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Flatten(),
            layers.Dense(latent_dimension, activation='relu'),
            layers.Dropout(0.2),
        ])

        self.mean_layer = layers.Dense(latent_dimension)
        self.variance_layer = layers.Dense(latent_dimension)

        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dimension, activation='relu'),
            layers.Dense(8*8*64, activation='relu'),
            layers.Reshape((8, 8, 64)),
            layers.Conv2DTranspose(64, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, kernel_size=(3, 3), activation='sigmoid', padding='same')
        ])

    def encode(self, input_images):
        encoded_images = self.encoder(input_images)
        mean = self.mean_layer(encoded_images)
        variance = self.variance_layer(encoded_images)
        return mean, variance

    def decode(self, latent_space):
        return self.decoder(latent_space)

    def reparameterize(self, mean, variance):
        epsilon = tf.random.normal(shape=mean.shape)
        z = mean + tf.exp(0.5 * variance) * epsilon
        return z

    def call(self, inputs):
        mean, variance = self.encode(inputs)
        z = self.reparameterize(mean, variance)
        output = self.decode(z)
        return output, mean, variance

# Compiling and training the model
vae = VariationalAutoEncoder()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function
def compute_loss(model, input_images):
    output, mean, variance = model(input_images)
    reconstruction_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(input_images, output))
    kl_divergence = -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))
    return reconstruction_loss + kl_divergence

@tf.function
def train_step(model, input_images, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, input_images)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Defining batch size
batch_size = 64

# Preparing training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)

num_epochs = 20
for epoch in range(num_epochs):
    start_time = time.time()
    for batch in train_dataset:
        train_step(vae, batch, optimizer)
    loss = tf.keras.metrics.Mean()
    for i in range(0, len(test_images), batch_size):
        test_image_batch = test_images[i:i+batch_size]
        loss(compute_loss(vae, test_image_batch))
    elbo = -loss.result()
    end_time = time.time()
    print('Epoch: {}, Test set ELBO: {}, '
          'time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
