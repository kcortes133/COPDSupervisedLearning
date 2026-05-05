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


# //TODO
# - PCA on different datasets
# - feature selection
# - distribution

def pca(data, color):
    p = PCA(2)
    projected = p.fit_transform(data)
    plt.scatter(projected[:,0], projected[:,1],
                edgecolor='none', alpha=0.5, c=color,
                #cmap=plt.cm.get_cmap('nipy_spectral', 10)
                )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.show()
    return


def recFeatSelec(X,y):
    svc = SVC(kernel='linear', C=1)
    rfe = RFE(estimator=svc, n_features_to_select=500)
    features = rfe.fit_transform(X,y)
    ranking = rfe.ranking_
    return features


def uniFeatureSelec(X, y, k):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    #print(selector.get_feature_names_out())
    #X_test = selector.transform(X_test)
    #X_new = SelectKBest(chi2, k=k).fit_transform(X,y)
    return X_new

def featureFilter(X, cutOff, title):
    num_rows, num_cols = X.shape
    vars = []
    for i in range(num_cols):
        col_var = np.std(X[:,i])/np.mean(X[:,i])
        vars.append(col_var)

    high = []
    count = 0
    for v in range(len(vars)):
        if vars[v] > cutOff:
            count+=1
            high.append(v)
    #print(count)

    return vars, high

def standardizeData(X, cutoff, title):
    num_rows, num_cols = X.shape
    new_X = np.zeros(X.shape)
    coeff = []
    high = []
    for i in range(num_cols):
        meani = np.mean(X[:,i])
        stdi = np.std(X[:,i])
        new_X[:,i] = abs(X[:,i]-meani)/stdi
        if stdi/meani > cutoff:
            high.append(i)

        coeff.append(stdi/meani)
    #plt.figure(figsize=(8, 6), dpi=80)
    #plt.rc('font', size=16)
    #plt.hist(coeff, rwidth=0.9, bins=40)
    #plt.ylabel('Number of features')
    #plt.xlabel('Coefficient of Variation')
    #plt.title(title)
    #plt.show()

    return new_X[:, high]


def getRidofGold1(clinGold, y):
    print(clinGold.shape)
    print(y.shape)
    rowDel = []
    c = 0
    for i in range(len(clinGold)):
        if clinGold[i] == 1:
            rowDel.append(i)
            c+=1

    clinGold = np.delete(clinGold, rowDel, axis=0)
    y = np.delete(y, rowDel, axis=0)

    print(clinGold.shape)
    print(y.shape)
    return clinGold, y

def lotsOfZeros(data):
    num_rows, num_cols = data.shape

    count = 0
    for row in data:
        numZ = np.count_nonzero(row)
        count+=1


def logTransform(X):
    transformer = FunctionTransformer(np.log1p)
    #transformer = FunctionTransformer(np.log2)
    X = transformer.transform(X)
    return X


def dataSpread(yTrain, yTest, title):
    ic,e, plot = plt.hist([yTrain, yTest], color=['skyblue', 'red'], label=['Train', 'Test'], bins=12)
    #plt.xticks([0,10,20, 30, 40, 50, 60], ha='center')
    print(ic)


    plt.ylabel('Number of patients')
    plt.xlabel(title)

    plt.legend()
    plt.title(title+ ' Patient Distribution')
    plt.show()

    return