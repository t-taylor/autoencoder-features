def get_metrics(model, X_test, Y_test):
    output = {}

    n = X_test.shape[0]
    y_pred = model.predict(X_test)
    inccorrect = (y_test != y_pred).sum()

    # Accuracy metric
    output['accuracy'] = 1 - (inccorrect / n)

    # TODO
    output['tpr'] = 'a'
    # TODO
    output['tnr'] = 'a'
    # TODO
    output['ppv'] = 'a'

