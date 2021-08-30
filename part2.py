### Part 2 testing, training, and experimentation

# * Training autoencoders
# * dimensional reduction with variance threshold and PCA 
# * all permutations with other ml algorithms

### Workflow
# * Step 1 dimensional reduction, Autoecoder, PCA with Autoecoder, PCA with variance threshold, or Nothing
# * Step 2, Choice of ML algorithm
# * Step 3, Calculate metrics

from part1 import nsl_multiclass, malware_df, multi_to_bin
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import csv
import numpy as np
import tensorflow as tf
import utils.dim_red as dr
import utils.eval as ev
import utils.ml as ml

def main():
  run_saves()

def save_models():
  print('Start nsl binary')

  (train, test) = multi_to_bin(nsl_multiclass())
  X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  Y_train_raw = train.values[:,-1]
  X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  Y_test_raw = test.values[:,-1]

  for dimred, model in dr.create_models(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw):
    print(dimred)
    model.save('models/nsl_bin/' + dimred)


  print('Start nsl multiclass')

  (train, test) = nsl_multiclass()
  X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  Y_train_raw = train.values[:,-1]
  X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  Y_test_raw = test.values[:,-1]

  for dimred, model in dr.create_models(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw):
    print(dimred)
    model.save('models/nsl_multi/' + dimred)

def run_saves():

  print('Start nsl binary')

  (train, test) = multi_to_bin(nsl_multiclass())
  X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  Y_train = train.values[:,-1]
  X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  Y_test = test.values[:,-1]

  inputs = dr.dimentional_reductions_from_saves(X_train_raw, X_test_raw, 'models/nsl_multi/')
  with open('nsl-binary-results.csv', 'a') as f:
    cw = csv.writer(f)
    cw.writerow(['dimred', 'modeltype', 'accuracy', 'precision', 'recall', 'f1', 'mcc'])
    for dimred, (X_train, X_test) in inputs:
      models = ml.generate_models(X_train, Y_train)

      for modeltype, model in models.items():
        metrics = ev.get_binary_metrics(model, X_test, Y_test)

        print(dimred, modeltype, metrics)
        cw.writerow([dimred, modeltype, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc']])

  #print('Start nsl multiclass')

  #(train, test) = nsl_multiclass()
  #X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  #Y_train = train.values[:,-1]
  #X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  #Y_test = test.values[:,-1]

  #inputs = dr.dimentional_reductions_from_saves(X_train_raw, X_test_raw, 'models/nsl_multi/')
  #with open('nsl-multiclass-results.csv', 'a') as f:
  #  cw = csv.writer(f)
  #  cw.writerow(['dimred', 'modeltype', 'accuracy', 'precision', 'recall', 'f1', 'mcc'])
  #  for dimred, (X_train, X_test) in inputs:
  #    models = ml.generate_models(X_train, Y_train)

  #    for modeltype, model in models.items():
  #      metrics = ev.get_multiclass_metrics(model, X_test, Y_test)

  #      print(dimred, modeltype, metrics)
  #      cw.writerow([dimred, modeltype, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc']])

def full_run():

  ## Malware Multiclass

  #print('Start malware multiclass')
  #maldf = malware_df()

  #X_raw = np.asarray(maldf.values[:,0:-1]).astype(np.float32)
  #Y_raw = maldf.values[:,-1]
  #X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X_raw, Y_raw, test_size=0.2)
  #inputs = dr.dimentional_reductions(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw)
  #with open('malware-multiclass-results.csv', 'wt') as f:
  #  cw = csv.writer(f)
  #  cw.writerow(['dimred', 'modeltype', 'accuracy', 'precision', 'recall', 'f1', 'mcc'])
  #  for dimred, (X_train, Y_train, X_test, Y_test) in inputs.items():
  #    models = ml.generate_models(X_train, Y_train)

  #    for modeltype, model in models.items():
  #      metrics = ev.get_multiclass_metrics(model, X_test, Y_test)

  #      print(dimred, modeltype, metrics)
  #      cw.writerow([dimred, modeltype, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc']])

  ## NSL Binary

  print('Start nsl binary')

  (train, test) = multi_to_bin(nsl_multiclass())
  X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  Y_train_raw = train.values[:,-1]
  X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  Y_test_raw = test.values[:,-1]

  inputs = dr.dimentional_reductions(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw)
  with open('nsl-binary-results.csv', 'wt') as f:
    cw = csv.writer(f)
    cw.writerow(['dimred', 'modeltype', 'accuracy', 'precision', 'recall', 'f1', 'mcc'])
    for dimred, (X_train, Y_train, X_test, Y_test) in inputs.items():
      models = ml.generate_models(X_train, Y_train)

      for modeltype, model in models.items():
        metrics = ev.get_binary_metrics(model, X_test, Y_test)

        print(dimred, modeltype, metrics)
        cw.writerow([dimred, modeltype, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc']])

  ## NSL Multiclass

  print('Start nsl multiclass')

  (train, test) = nsl_multiclass()
  X_train_raw = np.asarray(train.values[:,0:-1]).astype(np.float32)
  Y_train_raw = train.values[:,-1]
  X_test_raw = np.asarray(test.values[:,0:-1]).astype(np.float32)
  Y_test_raw = test.values[:,-1]

  inputs = dr.dimentional_reductions(X_train_raw, X_test_raw, Y_train_raw, Y_test_raw)
  with open('nsl-multiclass-results.csv', 'wt') as f:
    cw = csv.writer(f)
    cw.writerow(['dimred', 'modeltype', 'accuracy', 'precision', 'recall', 'f1', 'mcc'])
    for dimred, (X_train, Y_train, X_test, Y_test) in inputs.items():
      models = ml.generate_models(X_train, Y_train)

      for modeltype, model in models.items():
        metrics = ev.get_multiclass_metrics(model, X_test, Y_test)

        print(dimred, modeltype, metrics)
        cw.writerow([dimred, modeltype, metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['mcc']])

if __name__ == '__main__':
  main()
