import sklearn.metrics as met

def get_multiclass_metrics(model, X_test, Y_test):
    output = {}

    Y_pred = model.predict(X_test)

    output['accuracy'] = met.accuracy_score(Y_test, Y_pred)
    output['precision'] = met.precision_score(Y_test, Y_pred, average='weighted', zero_division=0)
    output['recall'] = met.recall_score(Y_test, Y_pred, average='weighted', zero_division=0)
    output['f1'] = met.f1_score(Y_test, Y_pred, average='weighted')
    output['mcc'] = met.matthews_corrcoef(Y_test, Y_pred)

    return output

def get_binary_metrics(model, X_test, Y_test):
    output = {}

    Y_pred = model.predict(X_test)

    output['accuracy'] = met.accuracy_score(Y_test, Y_pred)
    output['precision'] = met.precision_score(Y_test, Y_pred)
    output['recall'] = met.recall_score(Y_test, Y_pred)
    output['f1'] = met.f1_score(Y_test, Y_pred)
    output['mcc'] = met.matthews_corrcoef(Y_test, Y_pred)

    return output

