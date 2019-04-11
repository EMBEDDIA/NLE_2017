

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import normalize



class BM25Transformer(BaseEstimator, TransformerMixin):


    def __init__(self, norm='l2', smooth_idf=True, k=1.2, b=0.75):
        self.norm = norm
        self.smooth_idf = smooth_idf
        self.k = k
        self.b = b

    def fit(self, X):
        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        n_samples, n_features = X.shape

        if sp.isspmatrix_csr(X):
            df = np.bincount(X.indices, minlength=X.shape[1])
        else:
            df = np.diff(sp.csc_matrix(X, copy=False).indptr)

        df += int(self.smooth_idf)
        n_samples += int(self.smooth_idf)


        bm25idf = np.log((n_samples - df + 0.5) / (df + 0.5))
        self._idf_diag = sp.spdiags(bm25idf, diags=0, m=n_features, n=n_features)

        return self

    def transform(self, X, copy=True):

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            X = sp.csr_matrix(X, copy=copy)
        else:
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        D = (X.sum(1) / np.average(X.sum(1))).reshape((n_samples, 1))
        D = ((1 - self.b) + self.b * D) * self.k

        addition = sp.lil_matrix(X.shape)
        sparse = X.tocoo()
        for i, j, v in zip(sparse.row, sparse.col, sparse.data):
            addition[i, j] = v + D[i, 0]
        D_X = addition.tocsr()

        np.divide(X.data * (self.k + 1), D_X.data, X.data)

        if not hasattr(self, "_idf_diag"):
            raise ValueError("idf vector not fitted")
        expected_n_features = self._idf_diag.shape[0]
        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))
        X = X * self._idf_diag

        if self.norm:
            X = normalize(X, norm=self.norm, copy=False)

        return X


