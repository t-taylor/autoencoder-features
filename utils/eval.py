def get_metrics(model, X_test, Y_test):
    output = {}

    n = X_test.shape[0]
    Y_pred = model.predict(X_test)
    inccorrect = (Y_test != Y_pred).sum()

    # Accuracy metric
    output['accuracy'] = 1 - (inccorrect / n)

    # TODO
    output['tpr'] = 'a'
    # TODO
    output['tnr'] = 'a'
    # TODO
    output['ppv'] = 'a'

    return output

