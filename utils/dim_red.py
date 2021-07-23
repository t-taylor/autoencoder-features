import unittest as ut
import numpy as np
import utils.cvae as cvae
from sklearn import decomposition, preprocessing

def pca(X_train, X_test, n_components):

  pca_state = decomposition.PCA(n_components=n_components)
  pca_state.fit(X_train)

  X_train_pca = preprocessing.minmax_scale(pca_state.transform(X_train), axis=1)
  X_test_pca = preprocessing.minmax_scale(pca_state.transform(X_test), axis=1)

  return (X_train_pca, X_test_pca)

def dimentional_reductions(X_train, X_test, Y_train, Y_test):

  outputs = {}

  # No dimensional reduction
  outputs['none'] = (X_train, Y_train, X_test, Y_test)

  # Variational autoencoder
  #for e in range(50, 550, 50):
  #  for lf in range(2, 15):
  #for e in [50]:
  #  for lf in [4]:
  #    print('dm1', e, lf)
  #    (X_train_cvae_pca, X_test_cvae_pca) = cvae.apply_cvae(X_train, X_test, latent_dim=lf, epochs=e)
  #    outputs['autoencoder_' + str(e) + '_' + str(lf)] = (X_train_cvae_pca, Y_train, X_test_cvae_pca, Y_test)

  ## PCA with mle
  ## https://tminka.github.io/papers/pca/minka-pca.pdf
  #(X_train_pca_mle, X_test_pca_mle) = pca(X_train, X_test, 'mle')
  #outputs['pca_with_mle'] = (X_train_pca_mle, Y_train, X_test_pca_mle, Y_test)
  ##for e in range(50, 550, 50):
  ##  for lf in range(2, 15):
  #for e in [50]:
  #  for lf in [4]:
  #    print('dm2', e, lf)
  #    (X_train_cvae_pca, X_test_cvae_pca) = cvae.apply_cvae(X_train_pca_mle, X_test_pca_mle, latent_dim=lf, epochs=e)
  #    outputs['pca_with_varthresh_mle_autoencoder_' + str(e) + '_' + str(lf)] = (X_train_cvae_pca, Y_train, X_test_cvae_pca, Y_test)

  ##for n in range(80,100):
  #for n in [90]:
  #  thresh = n / 100
  #  # Principal component analysis with variance threshold
  #  (X_train_pca_thresh, X_test_pca_thresh) = pca(X_train, X_test, thresh)
  #  outputs['pca_with_varthresh_' + str(thresh)] = (X_train_pca_thresh, Y_train, X_test_pca_thresh, Y_test)

  #  #for e in range(50, 550, 50):
  #  #  for lf in range(2, 15):
  #  for e in [50]:
  #    for lf in [4]:
  #      print('dm3', e, lf, n)
  #      (X_train_cvae_pca, X_test_cvae_pca) = cvae.apply_cvae(X_train_pca_thresh, X_test_pca_thresh, latent_dim=lf, epochs=e)
  #      outputs['pca_with_varthresh_' + str(thresh) + '_autoencoder_' + str(e) + '_' + str(lf)] = (X_train_cvae_pca, Y_train, X_test_cvae_pca, Y_test)

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
