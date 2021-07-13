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
import utils.dim_red as dr
import utils.eval as ev
import utils.ml as ml

def demo():
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
        build_fn= ml.dense_model(inlen, outlen), epochs=20, batch_size=2, verbose=2)

    kfold = KFold(n_splits=10, shuffle=True)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def main():
    (train, test) = nsl_multiclass()
    train = train.values

    inputs = dr.dimentional_reductions(train, test)

    for dimred, (X_train, Y_train, X_test, Y_test) in inputs.items():
        models = ml.generate_models(X_train, Y_train)

        for modeltype, model in models.items():
            metrics = ev.get_metrics(model, X_test, Y_test)
            # Then write results to csv TODO

if __name__ == '__main__':
    main()
