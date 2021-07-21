import numpy as np
import utils.eval as eva
import utils.ml as ml
import part1
import unittest as ut
import utils.cvae as cvae

class top_level_tests(ut.TestCase):

  def test_accuracy(self):
    (train, test) = part1.nsl_multiclass()
    train = train.values
    X_train = np.asarray(train[:,0:-2]).astype(np.float32)
    test = test.values
    X_test = np.asarray(test[:,0:-2]).astype(np.float32)
    latent_dim = 2
    epochs = 7
    (X_train_cvae, X_test_cvae) = cvae.apply_cvae(X_train, X_test, latent_dim=latent_dim, epochs=epochs)

    Y_train = train[:,-2]
    Y_test = test[:,-2]

    model = ml.make_bayesian(X_train_cvae, Y_train)
    print(eva.get_metrics(model, X_test_cvae, Y_test))


if __name__ == '__main__':
  ut.main()
