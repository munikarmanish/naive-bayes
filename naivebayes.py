import numpy as np

class NaiveBayesClassifier:

    def _f(self, x, u, var):
        e = np.exp(-0.5 * (x-u)**2 / var)
        return e / np.sqrt(2 * np.pi * var)

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

        # Mean and variance
        self.u = {}
        self.var = {}
        for _class in X:
            self.u[_class] = X[_class].mean(0).T
            self.var[_class] = X[_class].var(0).T

        return self

    def predict(self, x):
        x = np.reshape(np.matrix(x), (self.n, 1))
        max_p, max_class = 0, list(self.u.keys())[0]
        for c in self.u:
            p = self.prior[c]
            for i in range(len(x)):
                p *= self._f(x[i], self.u[c][i], self.var[c][i])
            if p > max_p:
                max_p = p
                max_class = c
        return max_class

