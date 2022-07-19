from statistics import mean
import sklearn
from sklearn.decomposition import PCA
import numpy as np

import warnings

from math import log, sqrt
import numbers

import numpy as np
from scipy import linalg
from scipy.special import gammaln
from scipy.sparse import issparse
from scipy.sparse.linalg import svds


from spectrochempy.core.dataset.nddataset import NDDataset

# from spectrochempy.core.dataset.nddataset import NDDataset
# from spectrochempy.analysis import pca

def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
        out = np.cumsum(arr, axis=axis, dtype=np.float64)
        expected = np.sum(arr, axis=axis, dtype=np.float64)
        if not np.all(
            np.isclose(
                out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
        )
        ):
            warnings.warn(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum",
                RuntimeWarning,
        )
        return out

    
def svd_flip(u, v, u_based_decision=True):
        """
    u : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
    v : ndarray
        u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.
    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent."""
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
        # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u
    
def _infer_dimension(spectrum, n_samples):
    """Infers the dimension of a dataset with a given spectrum.
    The returned value will be in [1, n_features - 1].
    """
    ll = np.empty_like(spectrum)
    ll[0] = -np.inf  # we don't want to return n_components = 0
    for rank in range(1, spectrum.shape[0]):
        ll[rank] = _assess_dimension(spectrum, rank, n_samples)
    return ll.argmax()

class skpca(PCA):
    def __init__(self, n_components=None, *, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None) -> None:
        super().__init__(n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
    
    def fit(self, X, y=None) :
        if type(X) == NDDataset:
            return super().fit(X.data, y)
        else:
            return super().fit(X, y)
    
    def fit_transform(self, X, y= None):
        if type(X) == NDDataset:
            return super().fit_transform(X.data, y)
        else:
            return super().fit_transform(X, y)
    @property
    def score(self, X, y=None):
        output = NDDataset()
        output.title = "Score of the dataset"
        output.name = "Dataset score"
        output.data = super().score(X, y)
        #les dimensions de score sont pc1 et pc2 ?
        return output
    
    

    # Pour récupérer le loading, il faut calculer: loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    def _fit_full(self, X, n_components):
        """Fit the model by computing full SVD on X."""
        n_samples, n_features = X.shape

        if n_components == "mle":
            if n_samples < n_features:
                raise ValueError(
                    "n_components='mle' is only supported if n_samples >= n_features"
                )
        elif not 0 <= n_components <= min(n_samples, n_features):
            raise ValueError(
                "n_components=%r must be between 0 and "
                "min(n_samples, n_features)=%r with "
                "svd_solver='full'" % (n_components, min(n_samples, n_features))
            )
        elif n_components >= 1:
            if not isinstance(n_components, numbers.Integral):
                raise ValueError(
                    "n_components=%r must be of type int "
                    "when greater than or equal to 1, "
                    "was of type=%r" % (n_components, type(n_components))
                )

        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_

        U, S, Vt = linalg.svd(X, full_matrices=False)
        # flip eigenvectors' sign to enforce deterministic output
        U, Vt = svd_flip(U, Vt)

        components_ = Vt

        # Get variance explained by singular values
        explained_variance_ = (S**2) / (n_samples - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio_ = explained_variance_ / total_var
        singular_values_ = S.copy()  # Store the singular values.

        # Postprocess the number of components required
        if n_components == "mle":
            n_components = _infer_dimension(explained_variance_, n_samples)
        elif 0 < n_components < 1.0:
            # number of components for which the cumulated explained
            # variance percentage is superior to the desired threshold
            # side='right' ensures that number of features selected
            # their variance is always greater than n_components float
            # passed. More discussion in issue: #15669
            ratio_cumsum = stable_cumsum(explained_variance_ratio_)
            n_components = np.searchsorted(ratio_cumsum, n_components, side="right") + 1
        # Compute noise covariance using Probabilistic PCA model
        # The sigma2 maximum likelihood (cf. eq. 12.46)
        if n_components < min(n_features, n_samples):
            self.noise_variance_ = explained_variance_[n_components:].mean()
        else:
            self.noise_variance_ = 0.0

        self.n_samples_, self.n_features_ = n_samples, n_features
        self.components_ = components_[:n_components]
        self.n_components_ = n_components
        self.explained_variance_ = explained_variance_[:n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
        self.singular_values_ = singular_values_[:n_components]
        
        #Pour le mettre dans le dataset, il faut savoir quelle dimension est conservée.
        self.loadings_ = self.components_.T *np.sqrt(self.explained_variance_)

        return U, S, Vt
    

pca2 = skpca(n_components=2)
X = np.array([[1,2],[2,3]])
print(pca2.fit(X).mean_)
