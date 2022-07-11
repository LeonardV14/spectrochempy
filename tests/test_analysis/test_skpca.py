from numpy.testing import assert_allclose
import spectrochempy as scp

from spectrochempy.analysis import sk_learn_pca as sk_pca
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED

def test_pca_sk_learn(dataset):
    #test de format
    assert(type(dataset.explained_variance_ratio) == NDDataset, "Fail !")
    assert()
    
    return "Test done"