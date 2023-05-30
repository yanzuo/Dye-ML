import numpy as np
from sklearn.gaussian_process.kernels import Matern
from collections import defaultdict


class WrappedMatern(Matern):
    def __init__(self, param_types, length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=1.5):
        self.param_types = param_types
        super(WrappedMatern, self).__init__(length_scale, length_scale_bounds, nu)

    def _transform(self, X):
        # we collect the positions of the categorical variables in this dict
        categorical_group_indices = defaultdict(list)

        for i, ptype in enumerate(self.param_types):
            if ptype == 'continuous':
                pass  # do nothing
            elif ptype == 'discrete':
                X[:, i] = np.floor(X[:, i])
            else:
                categorical_group_indices[ptype].append(i)

        # set binary max for categorical groups
        for indices in categorical_group_indices.values():
            max_col = np.argmax(X[:, indices], axis=1)
            X[:, indices] = 0
            X[range(X.shape[0]), max_col] = 1

        return X

    def __call__(self, X, Y=None, eval_gradient=False):
        X = self._transform(X)

        if Y is not None:
            Y = self._transform(Y)

        return super(WrappedMatern, self).__call__(X, Y, eval_gradient)