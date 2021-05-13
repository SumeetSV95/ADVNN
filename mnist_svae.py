import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

tfd = tfp.distributions


class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=32, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=10)

        ])
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='relu'),
        ])

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def classify(self, z):
        y_hat = self.classifier(z)
        probs = tf.sigmoid(y_hat)
        return probs

    def discriminate(self, z):
        return self.discriminator(z)


def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def loadData():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = tf.one_hot(train_labels, 10)
    test_labels = tf.one_hot(test_labels, 10)
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)
    train_size = 60000
    batch_size = 32
    test_size = 10000
    datasetx = tf.data.Dataset.from_tensor_slices(train_images)
    datasety = tf.data.Dataset.from_tensor_slices(train_labels)
    train_dataset = tf.data.Dataset.zip((datasetx, datasety)).shuffle(train_size).batch(batch_size)

    datasetx = tf.data.Dataset.from_tensor_slices(test_images)
    datasety = tf.data.Dataset.from_tensor_slices(test_labels)
    test_dataset = tf.data.Dataset.zip((datasetx, datasety)).shuffle(test_size).batch(batch_size)

    return train_dataset, test_dataset


optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x, y):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    y_hat = model.classify(z)
    cce = tf.keras.losses.CategoricalCrossentropy()

    class_loss = cce(y, y_hat)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x + class_loss)


def computeLossDiscriminator(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    dist = tfd.Normal(loc=np.zeros(model.latent_dim), scale=np.ones(model.latent_dim))
    z_fake = dist.sample([len(x)])
    d_real = model.discriminate(z)
    d_fake = model.discriminate(z_fake)

    return -tf.reduce_mean(tf.math.log(tf.sigmoid(d_real)) + tf.math.log(1 - tf.sigmoid(d_fake)))


@tf.function
def train_step(model, x, optimizer, y):
    """Executes one training step and returns the loss.
  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


@tf.function
def train_discriminator_step(model, x, optimizer):
    """Executes one training step and returns the loss.
  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
    with tf.GradientTape() as tape:
        loss = computeLossDiscriminator(model, x)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def main():
    optimizer = tf.keras.optimizers.Adam(1e-4)
    train_dataset, test_dataset = loadData()
    epochs = 2
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 2
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)
    # generate_and_save_images(model, 0, test_sample)
    bestElbo = float('-inf')
    bestEpoch = -1
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, train_labels in train_dataset:
            train_step(model, train_x, optimizer, train_labels)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x, test_labels in test_dataset:
            loss(compute_loss(model, test_x, test_labels))
        elbo = -loss.result()
        if elbo>bestElbo:
            bestElbo = elbo
            bestEpoch = epoch
            model.save_weights("bestModels/weights_{}.h5".format(epoch))
        # display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, elbo, end_time - start_time))
        # generate_and_save_images(model, epoch, test_sample)
    print("loading weights...")
    #model.load_weights("bestModels/weights_{}.h5".format(bestEpoch))

    print("loading done...")
    for layer in model.encoder.layers:
        layer.trainable = False
    for layer in model.decoder.layers:
        layer.trainable = False
    for layer in model.classifier.layers:
        layer.trainable = False
    bestEpoch = -1
    bestloss = float('inf')
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x, train_labels in train_dataset:
            train_discriminator_step(model, train_x, optimizer)
        end_time = time.time()
        loss = tf.keras.metrics.Mean()
        for test_x, test_labels in test_dataset:
            loss(computeLossDiscriminator(model, test_x))
        currLoss = loss.result()
        if currLoss< bestloss:
            bestloss = currLoss
            bestEpoch = epoch
            model.save_weights("bestModels/weights_{}.h5".format(epoch))

        # display.clear_output(wait=False)
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
              .format(epoch, currLoss, end_time - start_time))


if __name__ == "__main__":
    main()
