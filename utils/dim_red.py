import unittest as ut
import numpy as np
import utils.cvae as cvae
from sklearn import decomposition

def pca(X_train, X_test, n_components):

  pca_state = decomposition.PCA(n_components=n_components)
  pca_state.fit(X_train)

  X_pca_train = pca_state.transform(X_train)
  X_pca_test = pca_state.transform(X_test)

  return (X_pca_train, X_pca_test)

def dimentional_reductions(train, test):

  X_train = np.asarray(train.values[:,0:-2]).astype(np.float32)
  Y_train = train.values[:,-2]

  X_test = np.asarray(test.values[:,0:-2]).astype(np.float32)
  Y_test = test.values[:,-2]

  outputs = {}

  # Principal component analysis with variance threshold TODO
  (X_pca_train, X_pca_test) = pca(X_train, X_test, 0.95)
  outputs['pca_with_varthresh'] = (X_pca_train, Y_train, X_pca_test, Y_test)

  ## Variational autoencoder TODO
  outputs['autoencoder'] = cvae.apply_cvae(X_train, X_test)

  ## Variational autoencoder with principal component analysis TODO
  #outputs['pca_with_autoencoder'] = 0

  # No dimensional reduction
  outputs['none'] = (X_train, Y_train, X_test, Y_test)

  return outputs

class dimentional_reductions_test(ut.TestCase):

  def test_pca(self):
    train = np.array([[0.387,4878, 5.42],
                      [0.723,12104,5.25],
                      [1,12756,5.52],
                      [1.524,6787,3.94]])
    test = np.array([[1,12756,5.52]])

    (pca_train, pca_test) = pca(train, test)
    test_out = np.array([[  4.25324997e+03,  -8.41288672e-01,  -8.37858943e-03],
                         [ -2.97275001e+03,  -1.25977271e-01,   1.82476780e-01],
                         [ -3.62475003e+03,  -1.56843494e-01,  -1.65224286e-01],
                         [  2.34425007e+03,   1.12410944e+00,  -8.87390454e-03]])

    round_mats = (lambda mat: np.vectorize(lambda x: round(x, 4))(mat))

    self.assertTrue((round_mats(pca_train) == round_mats(test_out)).all())
    test_Y = np.array([[ -3.62475003e+03,  -1.56843494e-01,  -1.65224286e-01]])
    self.assertTrue((round_mats(pca_test) == round_mats(test_Y)).all())

if __name__ == '__main__':
  ut.main()
