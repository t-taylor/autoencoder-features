import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.keras.utils import to_categorical

def to_numerical(xs):
  return list(map(lambda x: np.argmax(x),xs))

def lin_layers(indim, outdim, n_layers, n):
  grad = (outdim - indim) / n_layers
  return int(indim + (n + 1) * grad)

class Dense():
  def __init__(self, X, y, n_layers):

    self.encoder = prep.LabelEncoder()
    self.encoder.fit(y)

    encoded_y = self.encoder.transform(y)
    dummy_y = to_categorical(encoded_y)

    (_, indim) = X.shape
    dummy_y = to_categorical(encoded_y)
    (_, outdim) = dummy_y.shape

    self.outdim = outdim
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(input_shape=indim))

    self.model.add(tf.keras.layers.Dense(indim, activation='relu'))
    if n_layers > 1:
      for n in range(1, n_layers):
        self.model.add(tf.keras.layers.Dense(lin_layers(indim, outdim, n_layers, n), activation='relu'))

    self.model.add(tf.keras.layers.Dense(outdim, activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    self.model.fit(X, dummy_y, batch_size=128, epochs=200)
    self.model.summary()

  def predict(self, x):
    dummy_y_pred = self.model.predict(x)
    num_y_pred = to_numerical(dummy_y_pred)
    return self.encoder.inverse_transform(num_y_pred)

def create_dnet(X, y, n_layers):
    dmod = Dense(X, y, n_layers)

    return dmod
