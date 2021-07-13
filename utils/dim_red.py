import unittest as ut
import numpy as np
from sklearn import decomposition

def pca(X_train, X_test):

    pca_state = decomposition.PCA()
    pca_state.fit(X_train)

    X_pca_train = pca_state.transform(X_train)
    X_pca_test = pca_state.transform(X_test)

    return (X_pca_train, X_pca_test)

def dimentional_reductions(train, test):

    X_train = np.asarray(train[:,0:-2]).astype(np.float32)
    Y_train = train[:,-2]

    X_test = np.asarray(test[:,0:-2]).astype(np.float32)
    Y_test = test[:,-2]

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
