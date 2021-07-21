import sklearn.metrics as met

def get_metrics(model, X_test, Y_test):
    output = {}

    Y_pred = model.predict(X_test)

    output['accuracy'] = met.accuracy_score(Y_test, Y_pred)
    output['precision'] = met.precision_score(Y_test, Y_pred, average='weighted')
    output['recall'] = met.recall_score(Y_test, Y_pred, average='weighted')
    output['f1'] = met.f1_score(Y_test, Y_pred, average='weighted')

    return output

