from sklearn.decomposition import PCA

class PCA_implementation:
    def __init__(self, train_data, n_pc=0, var_per=0.95):
        self.train_data = train_data
        self.pca = PCA(n_pc) if n_pc>0 else PCA(var_per)
        self.pca.fit(train_data)

    def get_transformed_features(self, data):
        return self.pca.transform(data)

