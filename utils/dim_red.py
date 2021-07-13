def dimentional_reductions(train, test):
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
