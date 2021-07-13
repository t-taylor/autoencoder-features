### Part 2 testing, training, and experimentation
# * Training autoencoders
# * dimensional reduction with variance threshold and PCA 
# * all permutations with other ml algorithms

### Workflow
# * Step 1 dimensional reduction, Autoecoder, PCA with Autoecoder, PCA with variance threshold, or Nothing
# * Step 2, Choice of ML algorithm
# * Step 3, Calculate metrics

from part1 import nsl_multiclass, malware_df
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import utils.ml as ml

# A utility method to create a tf.data dataset from a Pandas Dataframe
def test_model(inlen, outlen):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inlen, input_dim=inlen, activation='relu'))
    model.add(tf.keras.layers.Dense(outlen, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lambda: model

def dimentional_reductions(train, test):
  methods = ['autoencoder', 'pca_with_autoencoder', 'pca_with_varthresh', 'none']
  outputs = {}
  for method in methods:
    if method == 'none':
      outputs[method] = (train, test)
    if method == 'pca_with_varthresh':
      # TODO
      outputs[method] = (train, test)
    if method == 'autoencoder':
      # TODO
      outputs[method] = (train, test)
    if method == 'pca_with_autoencoder':
      # TODO
      outputs[method] = (train, test)
  return outputs

def generate_models(train, test):
  methods = ['bayesian', 'svm', 'neural_net', 'decision_tree']
  outputs = {}
  for method in methods:
    if method == 'bayesian':
      # TODO
      outputs[method] == 'some_model'
    if method == 'svm':
      # TODO
      outputs[method] == 'some_model'
    if method == 'neural_net':
      # TODO
      outputs[method] == 'some_model'
    if method == 'decision_tree':
      # TODO
      outputs[method] == 'some_model'

def get_metrics(model, test):
  metrics = ['accuracy', 'tpr', 'tnr', 'ppv']
  output = {}
  for metric in metrics:
    if metric == 'accuracy':
      # TODO
      output[metric] = 'a'
    if metric == 'tpr':
      # TODO
      output[metric] = 'a'
    if metric == 'tnr':
      # TODO
      output[metric] = 'a'
    if metric == 'ppv':
      # TODO
      output[metric] = 'a'

def main():
    (train, test) = nsl_multiclass()
    train = train.values
    X = np.asarray(train[:,0:-2]).astype(np.float32)
    Y = train[:,-2]

    print('X.....')
    print(X)
    print('Y.....')
    print(Y)

    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    dummy_y = to_categorical(encoded_Y)
    (_, inlen) = X.shape
    (_, outlen) = dummy_y.shape

    estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(
      build_fn=test_model(inlen, outlen), epochs=20, batch_size=2, verbose=2)

    kfold = KFold(n_splits=10, shuffle=True)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

if __name__ == '__main__':
    main()
