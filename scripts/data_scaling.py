import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class MinMaxEpsScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.x_max = None
        self.x_min = None
        self.eps = eps

    def fit(self, X, y=None):
        self.x_min = np.min(X, axis=0)
        self.x_max = np.max(X, axis=0)
        return self

    def transform(self, X, y=None):
        X_norm = (X - self.x_min + self.eps) / (self.x_max - self.x_min + 2 * self.eps)
        mask = np.prod((X_norm > 0) & (X_norm < 1), axis=1).astype(bool)

        X_norm = X_norm[mask]
        if y is not None:
            y = y[mask]
            return X_norm, y
        return X_norm

    def inverse_transform(self, X, y=None):
        X_base = (self.x_max - self.x_min + 2 * self.eps) * X + self.x_min - self.eps
        return X_base


class LogitScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logit = np.log(X / (1 - X))
        no_infs_logit = np.ma.masked_invalid(logit)
        no_inf_max = no_infs_logit.max(axis=0).data
        no_inf_min = no_infs_logit.min(axis=0).data
        logit_clipped = np.clip(logit, no_inf_min, no_inf_max)
        if y is not None:
            return logit_clipped, y
        else:
            return logit_clipped

    def inverse_transform(self, X):
        X_exp = np.exp(X)
        return X_exp / (1 + X_exp)


class CATHODEScaler(BaseEstimator, TransformerMixin):
    def __init__(self, eps=0.0, use_standard=True):
        super().__init__()
        self.eps = eps
        self.use_standard = use_standard
        self.minmax = MinMaxEpsScaler(self.eps)
        self.logit = LogitScaler()
        if use_standard:
            self.standard = StandardScaler()

    def fit(self, X, y=None):
        self.minmax = self.minmax.fit(X, y)
        self.logit = self.logit.fit(X, y)
        if self.use_standard:
            X = self.minmax.transform(X)
            X = self.logit.transform(X)
            self.standard = self.standard.fit(X)
        return self

    def transform(self, X, y=None):
        if y is not None:
            X, y = self.minmax.transform(X, y)
            X, y = self.logit.transform(X, y)
        else:
            X = self.minmax.transform(X)
            X = self.logit.transform(X)
        if self.use_standard:
            X = self.standard.transform(X)
        if y is not None:
            return X, y
        else:
            return X

    def inverse_transform(self, X, y=None):
        if self.use_standard:
            X = self.standard.inverse_transform(X)
        X = self.logit.inverse_transform(X)
        X = self.minmax.inverse_transform(X)
        return X
