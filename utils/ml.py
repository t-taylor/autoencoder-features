import sklearn.naive_bayes as nb
import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.datasets as skdata
import sklearn.model_selection as skms
import tensorflow as tf
import unittest as ut
import utils.dense_net as den

def make_bayesian(X_train, Y_train):
  gnb = nb.GaussianNB()
  return gnb.fit(X_train, Y_train)

def make_svms(X_train, Y_train):
  kerns = ['rbf']
  out = {}
  for k in kerns:
    print('creating svm', k)
    msvm =  svm.SVC(kernel=k, gamma='auto') if kerns == 'poly' else svm.SVC(kernel=k)
    out[k] = msvm.fit(X_train, Y_train)
  return out

def make_dtrees(X_train, Y_train):
  criterion = ['gini', 'entropy']
  splitter = ['best', 'random']
  out = {}
  for c in criterion:
    for s in splitter:
      for mss in range(2, 10, 2):
        for msl in range(1, 5):
          print('creating dtree', c, s, mss, msl)
          mdt = tree.DecisionTreeClassifier(criterion=c, splitter=s, min_samples_split=mss, min_samples_leaf=msl)
          out['dtree_' + str(c) + '_' + str(s) + '_' + str(mss) + '_' + str(msl)] = mdt.fit(X_train, Y_train)
  return out

def make_neural_nets(X_train, Y_train):
  out = {}

  for n_layers in [1, 3, 5]:
    model = den.create_dnet(X_train, Y_train, n_layers)
    out['nn_' + str(n_layers)] = model

  return out


def generate_models(X_train, Y_train):
  outputs = {}
  # Gaussian naive bayes
  #outputs['bayesian'] = make_bayesian(X_train, Y_train)

  # Support vector machine
  #for kern, m in make_svms(X_train, Y_train).items():
  #  outputs['svm_' + str(kern)] = m

  ## Decision Trees
  #outputs = outputs | make_dtrees(X_train, Y_train)

  # Neural Networks
  outputs = outputs | make_neural_nets(X_train, Y_train)

  return outputs

class generate_models_test(ut.TestCase):

  def test_bayesian(self):
    X, y = skdata.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=0)

    model = make_bayesian(X_train, y_train)
    y_pred = model.predict(X_test)
    wrong = (y_test != y_pred).sum()
    self.assertEqual(wrong, 4)

  def test_svm(self):
    X, y = skdata.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=0)

    model = make_svms(X_train, y_train)['rbf']
    y_pred = model.predict(X_test)
    wrong = (y_test != y_pred).sum()
    self.assertEqual(wrong, 4)

  def test_dtrees(self):
    X, y = skdata.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=0)

    models = make_dtrees(X_train, y_train)
    for l, m in models.items():
      y_pred = m.predict(X_test)
      wrong = (y_test != y_pred).sum()

if __name__ == '__main__':
  ut.main()
