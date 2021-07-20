import numpy as np
import part1
import unittest as ut
import utils.cvae as cvae

class top_level_tests(ut.TestCase):
  def test_partial_data(self):
    (train, test) = part1.nsl_multiclass()
    train = train.values
    X_train = np.asarray(train[:,0:-2]).astype(np.float32)
    print('more than one', np.vectorize(lambda x: x >= 0 and x <= 1)(X_train).sum().sum())
    print('more than one', np.vectorize(lambda x: True)(X_train).sum().sum())
    test = test.head(n=32).values
    X_test = np.asarray(test[:,0:-2]).astype(np.float32)
    latent_dim = 2
    epochs = 7
    (X_train_cvae, X_test_cvae) = cvae.apply_cvae(X_train, X_test, latent_dim=latent_dim, epochs=epochs)

if __name__ == '__main__':
  ut.main()
