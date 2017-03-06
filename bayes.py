import numpy as np

class BayesClassifier:

    def _f(self, x, u, cov):
        e = np.exp(-0.5 * (x-u).T * np.linalg.pinv(cov) * (x-u))
        return e / np.sqrt(np.linalg.det(2*np.pi*cov))


    def train(self, dataset):
        # Pre-processing
        dataset = np.matrix(dataset)
        dataset_x = np.matrix(dataset[:,:-1], dtype='float')
        dataset_y = np.matrix(dataset[:,-1])
        self.m, self.n = dataset_x.shape
        X = {}
        count = {}
        for i in range(self.m):
            _class = dataset_y[i,0]
            if _class not in X:
                X[_class] = dataset_x[i]
                count[_class] = 1
            else:
                X[_class] = np.vstack((X[_class], dataset_x[i]))
                count[_class] += 1

        # Also set the prior probabilities
        self.prior = {}
        for _class in count:
            self.prior[_class] = count[_class] / self.m

        # Mean
        self.u = {}
        for _class in X:
            self.u[_class] = X[_class].mean(0).T

        # Covariance matrix
        Z = {}
        self.cov = {}
        for _class in X:
            Z[_class] = X[_class] - self.u[_class].T
            self.cov[_class] = (Z[_class].T * Z[_class]) / count[_class]

        return self

    def predict(self, x):
        x = np.reshape(np.matrix(x), (self.n, 1))
        max_p, max_class = 0, list(self.u.keys())[0]
        for c in self.u:
            p = self._f(x, self.u[c], self.cov[c]) * self.prior[c]
            if p > max_p:
                max_p = p
                max_class = c
        return max_class

