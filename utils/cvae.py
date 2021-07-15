# Code adapted from https://www.tensorflow.org/tutorials/generative/cvae
import math
import numpy as np
import random as r
import tensorflow as tf
import time
import unittest as ut

def batch(d, bin_size):
  data = d.copy()
  np.random.shuffle(data)
  data = np.array_split(data, math.ceil(len(data)/bin_size))
  return np.array(data)


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, feature_dim):
    super(CVAE, self).__init__()
    print('VALES__', latent_dim, feature_dim)
    self.latent_dim = latent_dim
    self.optimizer = tf.keras.optimizers.Adam(1e-4)
    self.encoder = tf.keras.Sequential(
      [tf.keras.layers.InputLayer(input_shape=feature_dim),
       tf.keras.layers.Dense(feature_dim, activation='relu'),
       tf.keras.layers.Dense(int((latent_dim + feature_dim) / 2), activation='relu'),
       # No activation
       tf.keras.layers.Dense(latent_dim + latent_dim)])

    self.decoder = tf.keras.Sequential(
      [tf.keras.layers.InputLayer(input_shape=latent_dim),
       tf.keras.layers.Dense(latent_dim, activation='relu'),
       tf.keras.layers.Dense(int((latent_dim + feature_dim) / 2), activation='relu'),
       # No activation
       tf.keras.layers.Dense(feature_dim)])

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    pred = self.encoder(x)
    mean, logvar = tf.split(pred, num_or_size_splits=2, axis=1)
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

  @tf.function
  def train_step(self, x):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
      loss = compute_loss(self, x)
    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))



def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent)
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def train_model(model, X_train, latent_dim, epochs):
  for epoch in range(epochs):
    start_time = time.time()
    print('Starting epoch', epoch)
    X_train_batched = batch(X_train, 32)
    for x in X_train_batched:
      model.train_step(x)
    end_time = time.time()
    loss = tf.keras.metrics.Mean()
    for test_x in X_train_batched:
      loss(compute_loss(model, test_x))
    elbo = -loss.result()
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))

def apply_cvae(X_train, X_test, latent_dim=5, epochs=100):
  # Get model
  model = CVAE(latent_dim, X_train.shape[1])
  # Train model on X_train
  print('Starting CVAE training at', time.time())
  train_model(model, X_train, latent_dim, epochs)
  print('Finishing CVAE training at', time.time())
  # Apply encoder on X_train && X_test
  X_train_cvae, _ = model.encode(X_train)
  X_test_cvae, _ = model.encode(X_test)

  return (X_train_cvae, X_test_cvae)

class cvae_test(ut.TestCase):

  def test_train_model(self):
    X_train = np.array([[r.random() for i in range(8)] for j in range(49)]).astype(np.float32)
    X_test = np.array([X_train[0]])
    latent_dim = 2
    epochs = 9

    # Get model
    model = CVAE(latent_dim, 8)
    print(model.encoder.summary())
    print('data shape', X_train.shape)
    # Train model on X_train
    train_model(model, X_train, latent_dim, epochs)
    # Apply encoder on X_train && X_test

    X_train_cvae, _ = model.encode(X_train)
    X_test_cvae, _ = model.encode(X_test)
    #(train_c, test_c) = apply_cvae(x_train, x_test, latent_dim=2, epochs=2)

    #print(train_c)
    #print(test_c)

  def test_batch(self):
    X = np.array(range(20))
    Xb = batch(X, 3)
    print(X)
    print(Xb)

    self.assertEqual(Xb.shape, (7,))

  def test_apply_cvae(self):
    print('')
    X = np.array([[r.random() for i in range(8)] for j in range(128)]).astype(np.float32)
    X_test = np.array([[r.random() for i in range(8)] for j in range(32)]).astype(np.float32)
    latent_dim = 2
    epochs = 20

    (X_train_cvae, X_test_cvae) = apply_cvae(X, X_test, latent_dim=latent_dim, epochs=epochs)
    print(type(X_train_cvae))


if __name__ == '__main__':
  ut.main()
