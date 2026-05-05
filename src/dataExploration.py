"""
Feature engineering, dimensionality reduction, and basic EDA helpers.

This module is the main "data-prep" toolbox used by `main.py`:

    pca               — 2-component PCA scatter plot.
    recFeatSelec      — recursive feature elimination with a linear SVC
                        (currently unused by main.py, kept for experimentation).
    uniFeatureSelec   — SelectKBest(chi2) — the workhorse for picking the
                        top-k features per modality before classification.
    featureFilter     — coefficient-of-variation filter that returns the
                        (vars, high-CV-indices) pair (currently unused).
    standardizeData   — per-feature z-score, then drop columns whose
                        coefficient of variation is below `cutoff`.
                        Returns only the surviving columns.
    getRidofGold1     — drop rows whose label equals GOLD-1 (the ambiguous
                        intermediate stratum) from both label and feature
                        arrays in lockstep.
    lotsOfZeros       — sparsity diagnostic stub (no return value yet).
    logTransform      — `log(1+x)` element-wise.
    dataSpread        — train/test label-distribution histogram.

All routines accept numpy arrays unless noted.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.feature_selection import chi2, RFE
from sklearn.svm import SVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split


def pca(data, color):
    """Plot a 2-component PCA scatter coloured by `color`.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
    color : array-like, shape (n_samples,)
        Per-sample colour value (e.g. a label vector).
    """
    p = PCA(2)
    projected = p.fit_transform(data)
    plt.scatter(projected[:, 0], projected[:, 1],
                edgecolor='none', alpha=0.5, c=color,
                # cmap=plt.cm.get_cmap('nipy_spectral', 10)
                )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    return


def recFeatSelec(X, y):
    """Recursive feature elimination with a linear-kernel SVC.

    Picks the top 500 features by RFE ranking. Currently unused by `main.py`
    but kept available as an alternative to `uniFeatureSelec`.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)

    Returns
    -------
    numpy.ndarray, shape (n_samples, 500)
    """
    svc = SVC(kernel='linear', C=1)
    rfe = RFE(estimator=svc, n_features_to_select=500)
    features = rfe.fit_transform(X, y)
    ranking = rfe.ranking_
    return features


def uniFeatureSelec(X, y, k):
    """Univariate feature selection: top-k features by chi-squared score.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Must be non-negative — chi^2 cannot be computed on signed values.
        Upstream of this call, features are log-transformed and the
        absolute z-score is taken in `standardizeData`.
    y : array-like, shape (n_samples,)
        Class labels (binary in this pipeline).
    k : int
        Number of features to keep.

    Returns
    -------
    numpy.ndarray, shape (n_samples, k)
    """
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new


def featureFilter(X, cutOff, title):
    """Coefficient-of-variation filter.

    For each column, compute std/mean and report which columns exceed
    `cutOff`. Currently unused; superseded in the pipeline by
    `standardizeData` (which folds the same filter into a z-score).

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    cutOff : float
        CV threshold; columns with CV > cutOff are flagged.
    title : str
        Diagnostic label (unused — present for parity with `standardizeData`).

    Returns
    -------
    (vars, high) : tuple
        vars : list[float]   per-column CV values
        high : list[int]     column indices whose CV exceeds `cutOff`
    """
    num_rows, num_cols = X.shape
    vars = []
    for i in range(num_cols):
        col_var = np.std(X[:, i]) / np.mean(X[:, i])
        vars.append(col_var)

    high = []
    count = 0
    for v in range(len(vars)):
        if vars[v] > cutOff:
            count += 1
            high.append(v)

    return vars, high


def standardizeData(X, cutoff, title):
    """Per-feature z-score, then keep only high-CV columns.

    For each column:
        new_X[:, i] = |X[:, i] - mean_i| / std_i

    The result is the *absolute* z-score, which is non-negative and so safe
    for the downstream chi^2 feature selector. Columns whose coefficient of
    variation (std/mean) does not exceed `cutoff` are dropped — they carry
    little signal relative to their mean.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
    cutoff : float
        Coefficient-of-variation threshold; columns with CV > cutoff are kept.
    title : str
        Diagnostic label used by the (commented-out) histogram block.

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_kept_features)
    """
    num_rows, num_cols = X.shape
    new_X = np.zeros(X.shape)
    coeff = []
    high = []
    for i in range(num_cols):
        meani = np.mean(X[:, i])
        stdi = np.std(X[:, i])
        new_X[:, i] = abs(X[:, i] - meani) / stdi
        if stdi / meani > cutoff:
            high.append(i)

        coeff.append(stdi / meani)

    # Diagnostic histogram of per-column CVs — left commented for ad-hoc use:
    # plt.figure(figsize=(8, 6), dpi=80)
    # plt.rc('font', size=16)
    # plt.hist(coeff, rwidth=0.9, bins=40)
    # plt.ylabel('Number of features')
    # plt.xlabel('Coefficient of Variation')
    # plt.title(title)
    # plt.show()

    return new_X[:, high]


def getRidofGold1(clinGold, y):
    """Drop GOLD-1 rows from both the label vector and a feature matrix.

    GOLD-1 is treated as an ambiguous intermediate stratum and excluded from
    classification (the pipeline only contrasts GOLD 0 vs GOLD 2-4).

    Parameters
    ----------
    clinGold : numpy.ndarray, shape (n_samples,)
        GOLD-stage labels (0, 1, 2, 3, 4).
    y : numpy.ndarray, shape (n_samples, ...)
        Feature matrix (or any per-subject array) to drop the same rows from.

    Returns
    -------
    (clinGold_filtered, y_filtered) : tuple of numpy.ndarray
        Both with the GOLD-1 rows removed.
    """
    print(clinGold.shape)
    print(y.shape)
    rowDel = []
    c = 0
    for i in range(len(clinGold)):
        if clinGold[i] == 1:
            rowDel.append(i)
            c += 1

    clinGold = np.delete(clinGold, rowDel, axis=0)
    y = np.delete(y, rowDel, axis=0)

    print(clinGold.shape)
    print(y.shape)
    return clinGold, y


def lotsOfZeros(data):
    """Stub: counts non-zero entries per row.

    Currently a placeholder — the loop runs but neither `numZ` nor `count`
    is exported. Kept as a reminder for the sparsity-screen experiment that
    was never finished.
    """
    num_rows, num_cols = data.shape

    count = 0
    for row in data:
        numZ = np.count_nonzero(row)
        count += 1


def logTransform(X):
    """Element-wise `log(1+x)` via `FunctionTransformer(np.log1p)`.

    Used on proteomics + metabolomics — both have heavy-tailed abundance
    distributions that compress nicely under log1p without blowing up at 0.
    """
    transformer = FunctionTransformer(np.log1p)
    # transformer = FunctionTransformer(np.log2)
    X = transformer.transform(X)
    return X


def dataSpread(yTrain, yTest, title):
    """Plot a histogram of train vs test label distributions.

    Useful to sanity-check that train/test splits aren't drastically
    imbalanced for a continuous target.

    Parameters
    ----------
    yTrain, yTest : array-like
    title : str
        Used both as x-axis label and as part of the figure title.
    """
    ic, e, plot = plt.hist([yTrain, yTest],
                           color=['skyblue', 'red'],
                           label=['Train', 'Test'], bins=12)
    print(ic)

    plt.ylabel('Number of patients')
    plt.xlabel(title)
    plt.legend()
    plt.title(title + ' Patient Distribution')
    plt.show()
    return
