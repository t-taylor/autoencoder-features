import sklearn.naive_bayes as nb
import sklearn.datasets as skdata
import sklearn.model_selection as skms
import tensorflow as tf
import unittest as ut

def dense_model(inlen, outlen):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(inlen, input_dim=inlen, activation='relu'))
    model.add(tf.keras.layers.Dense(outlen, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return lambda: model

def bayesian(X_train, Y_train):
    gnb = nb.GaussianNB()
    return gnb.fit(X_train, Y_train)

def generate_models(X_train, Y_train):
    outputs = {}
    # Gaussian naive bayes
    outputs['bayesian'] = bayesian(X_train, Y_train)

    ## Support vector machine TODO
    #outputs['svm'] = 'some_model'

    ## TODO
    #outputs['neural_net'] = 'some_model'

    ## TODO
    #outputs['decision_tree'] = 'some_model'
    return outputs

class generate_models_test(ut.TestCase):

    def test_baysian(self):
        X, y = skdata.load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = skms.train_test_split(X, y, test_size=0.5, random_state=0)

        model = bayesian(X_train, y_train)
        y_pred = model.predict(X_test)
        wrong = (y_test != y_pred).sum()
        self.assertEqual(wrong, 4)

if __name__ == '__main__':
    ut.main()
