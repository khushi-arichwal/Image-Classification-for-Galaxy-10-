#importing necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses, optimizers
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizing pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Defining the autoencoder model
class Autoencoder(models.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same')
        ])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(32, (2, 2), strides=2, activation='relu', padding='same'),
            layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Compiling and training the model
autoencoder = Autoencoder()
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_images, train_images, epochs=10, shuffle=True, validation_data=(test_images, test_images))

# Evaluating the model
test_loss = autoencoder.evaluate(test_images, test_images)
print('Test loss:', test_loss)

# Displaying a few test images with reconstructions
num_images = 4
plt.figure(figsize=(10, 6))
for i in range(num_images):
    # Display original
    ax = plt.subplot(2, num_images, i + 1)
    plt.imshow(test_images[i])
    plt.title(f"original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, num_images, i + 1 + num_images)
    plt.imshow(autoencoder(test_images[i][np.newaxis])[0])
    plt.title(f"reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
