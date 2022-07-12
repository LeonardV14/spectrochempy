from numpy.testing import assert_allclose
import spectrochempy as scp
import numpy as np

from spectrochempy.utils.testing import assert_array_almost_equal

from spectrochempy.optional import import_optional_dependency

from spectrochempy.analysis import sk_learn_pca as pca
from sklearn.decomposition import PCA as sklPCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED

def test_pca_sk_learn(dataset):
    try:
        import_optional_dependency("scikit-learn")
    except ImportError:
        return

    #Lorsque l'input est un NDDataset 
    dataset = NDDataset.read("irdata/nh4y-activation.spg") 
    X = dataset.copy().data

    pcas = sklPCA(n_components=5, svd_solver="full")
    pcas.fit(X)

    pca = pca(n_components = 5, svd_solver = "full")
    pca.fit(dataset)

    assert_array_almost_equal(pca.singular_values_[:5], pcas.singular_values_[:5], 4)
    assert_array_almost_equal(
        pca.explained_variance_ratio_[:5], pcas.explained_variance_ratio_[:5] * 100.0, 4
    )



    #Lorsque l'input est un array
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    
    pcas = sklPCA(n_components=2) 
    pcas.fit(X)

    pca = pca(n_components = 2)
    pca.fit(X)

    assert_array_almost_equal(pca.singular_values_, pcas.singular_values_)