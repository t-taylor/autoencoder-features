import tensorflow as tf
import numpy as np
import sklearn.preprocessing as prep
from tensorflow.keras.utils import to_categorical

def to_numerical(xs):
  return list(map(lambda x: np.argmax(x),xs))

class Dense():
  def __init__(self, X, y, n_layers):

    (_, indim) = X.shape
    dummy_y = to_categorical(encoded_Y)
    (_, outdim) = dummy_y.shape

    self.outdim = outdim
    self.model = tf.keras.Sequential()
    self.model.add(tf.keras.layers.InputLayer(input_shape=indim))

    for n in n_layers:
      self.model.add(tf.keras.layers.Dense(indim, activation='relu'))

    self.model.add(tf.keras.layers.Dense(outdim, activation='softmax'))
    self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    self.encoder = prep.LabelEncoder()


  def train(self, X_train, Y_train):

    self.encoder.fit(Y)
    encoded_Y = self.encoder.transform(Y)
    dummy_y = to_categorical(encoded_Y)

    estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(
      build_fn= (lambda: self.model), epochs=20, batch_size=128, verbose=2)

  def predict(self, x):
    dummy_y_pred = self.predict(x)
    self.encoder.inverse_transform(to_numerical(dummy_y_pred))

def create_dnet(X, y, n_layers):
    dmod = Dense(indim, outdim, n_layers)
    dmod.train(X, y)

    return dmod
