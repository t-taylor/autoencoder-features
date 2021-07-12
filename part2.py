### Part 2 testing, training, and experimentation
# * Training autoencoders
# * dimensional reduction with variance threshold and PCA 
# * all permutations with other ml algorithms

from part1 import nsl_multiclass, malware_df
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import utils.ml as ml

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('class')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a StringLookup layer which will turn strings into integer indices
  if dtype == 'string':
    index = preprocessing.StringLookup(max_tokens=max_tokens)
  else:
    index = preprocessing.IntegerLookup(max_tokens=max_tokens)
  # Prepare a Dataset that only yields our feature
  feature_ds = dataset.map(lambda x, y: x[name])
  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)
  # Create a Discretization for our integer indices.
  encoder = preprocessing.CategoryEncoding(num_tokens=index.vocabulary_size())
  # Apply one-hot encoding to our indices. The lambda function captures the
  # layer so we can use them, or include them in the functional model later.
  return lambda feature: encoder(index(feature))

def test_model(inlen, outlen):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inlen, input_dim=inlen, activation='relu'))
    model.add(tf.keras.layers.Dense(outlen, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lambda: model

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

    estimator = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=test_model(inlen, outlen), epochs=200, batch_size=5, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    #train_ds = df_to_dataset(train)
    #test_ds = df_to_dataset(test)

    #feature_cols = []
    #for col in list(train.columns):
    #    if 'class' not in col or 'difficulty' not in col:
    #        feature_cols.append(tf.feature_column.numeric_column(col))

    #model = ml.dense_network(feature_cols)

    #model.fit(train_ds,
    #      epochs=10)

if __name__ == '__main__':
    main()
