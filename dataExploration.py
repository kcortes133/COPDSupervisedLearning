import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest


# //TODO
# - PCA on different datasets
# - feature selection
# - distribution

def pca(data):
    p = PCA(2)
    projected = p.fit_transform(data)
    plt.scatter(projected[:,0], projected[:,1],
                edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('nipy_spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    return


def uniFeatureSelec(data):
    selec = SelectKBest(chi2, k=3000).fit_transform(data)

    return

