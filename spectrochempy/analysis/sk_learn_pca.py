#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import spectrochempy as scp
import sklearn

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.analysis import pca

class sk_pca2():
    def __init__(self, dataset, n_components, values = None):
        self.dataset = dataset
        self.values = dataset._data
        self.n_components = n_components
    
    # Private methods
    
    def _explained_variance_ratio(self):
        return (PCA(self.n_components).fit(self.values).explained_variance_ratio_)
    
    def _singular_values(self):
        return (PCA(self.n_components).fit(self.values).singular_values_)
    
    def _explained_variance(self):
        return (PCA(self.n_components).fit(self.values).explained_variance_)
    
    def _mean(self):
        return (PCA(self.n_components).fit(self.values).mean_)
    
    def _noise_variance(self):
        return (PCA(self.n_components).fit(self.values).noise_variance_)
    
    def _n_samples(self):
        return (PCA(self.n_components).fit(self.values).n_samples_)
    
    def _components(self):
        return (PCA(self.n_components).fit(self.values).components_)
    
    def _n_components(self):
        return (PCA(self.n_components).fit(self.values).n_components_)
    
    def _n_features(self):
        return (PCA(self.n_components).fit(self.values).n_features_)
    
    def _n_features_in(self):
        return (PCA(self.n_components).fit(self.values).n_features_in_)
    
    def _features_names_in(self):
        return (PCA(self.n_components).fit(self.values).features_names_in_)
    
    # Public Methods
    @property
    def explained_variance_ratio(self):
        output = NDDataset()
        output.title = "Explained variance ratio of" + self.dataset.name
        output.name = "PCA explained variance ratio"
        output.data = self._explained_variance_ratio()
        return output
    
    @property
    def mean(self):
        output = NDDataset()
        output.title = "Mean of" + self.dataset.name
        output.name = "PCA mean"
        output.data = self._mean()
        return output
    
    @property
    def singular_values(self):
        output = NDDataset()
        output.title = "Singular values of" + self.dataset.name
        output.name = "PCA singular values"
        output.data = self._singular_values()
        return output
    
    @property
    def explained_variance(self):
        output = NDDataset()
        output.title = "Explained variance of" + self.dataset.name
        output.name = "PCA explained variance"
        output.data = self._explained_variance()
        return output
    
    @property
    def noise_variance(self):
        output = NDDataset()
        output.title = "Noise variance_ of" + self.dataset.name
        output.name = "PCA noise variance_"
        output.data = self._noise_variance()
        return output
    
    @property
    def n_samples(self):
        output = NDDataset()
        output.title = "n_samples_ of" + self.dataset.name
        output.name = "PCA n_samples"
        output.data = self._n_samples()
        return output
    
    @property
    def components(self):
        output = NDDataset()
        output.title = "Components of" + self.dataset.name
        output.name = "PCA components"
        output.data = self._components()
        return output
    
    @property
    def n_components(self):
        output = NDDataset()
        output.title = "N components of" + self.dataset.name
        output.name = "PCA n_components"
        output.data = self._n_components()
        return output
    
    @property
    def n_features(self):
        output = NDDataset()
        output.title = "N features of" + self.dataset.name
        output.name = "PCA N features"
        output.data = self._n_features()
        return output
    
    @property
    def n_features_in(self):
        output = NDDataset()
        output.title = "Number of features seen during the fit of" + self.dataset.name
        output.name = "PCA N features during the fit"
        output.data = self._n_features_in()
        return output
    
    @property
    def features_names_in(self):
        output = NDDataset()
        output.title = "Names of features seen during the fit of" + self.dataset.name
        output.name = "PCA N features names"
        output.data = self._features_names_in()
        return output
    
    # PCA
    
    def pcap(self):
        pca = PCA(self.n_components).fit_transform(self.values)
        plt.figure()
        plt.scatter(pca[:,0], pca[:,1])
        plt.xlabel("First PC")
        plt.ylabel("Second PC")


# In[92]:


dataset = scp.download_iris()
dataset._data


# In[93]:


dataset = scp.download_iris()
dataset


# In[ ]:




