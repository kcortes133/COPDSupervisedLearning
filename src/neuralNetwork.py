"""
Classifier wrappers used by the COPD pipeline.

The headline routine is `votingClassifier` — a soft-voting ensemble over
logistic-regression-with-elastic-net, an MLP, and a probabilistic SVC. It is
the only function in this module that `main.py` actually calls; the others
are kept for ad-hoc experimentation.

Public surface:

    rFC                          — fit a RandomForest (utility, unused).
    xgb                          — fit an XGBoost classifier and print
                                   accuracy on a held-out split (used by
                                   `test.py` only).
    votingClassifier             — full evaluation routine for the soft-voting
                                   ensemble: 5-fold CV scores, ROC, confusion
                                   matrix, prediction-distribution plots, and
                                   per-classifier accuracy bars.
    compute_feature_importance   — pull per-estimator coefficient magnitudes
                                   from a fitted VotingClassifier (called via
                                   commented-out hook inside votingClassifier).

NOTE: this module relies on `sklearn.metrics.plot_confusion_matrix` and
`sklearn.metrics.plot_roc_curve`, both removed in scikit-learn 1.2. Pin
scikit-learn < 1.2 (see requirements.txt).
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, cross_val_predict, StratifiedKFold,
)
from sklearn.metrics import (
    accuracy_score, confusion_matrix, plot_confusion_matrix,
    RocCurveDisplay, auc,
)
from sklearn import metrics
from sklearn.preprocessing import FunctionTransformer

from xgboost import XGBClassifier

import dataExploration
import plots


def rFC(X, Y):
    """Fit a small RandomForest (18 trees) and return the fitted classifier.

    Utility wrapper; not called by `main.py`.

    Parameters
    ----------
    X, Y : array-like
    """
    clf = RandomForestClassifier(n_estimators=18)
    clf = clf.fit(X, Y)
    return clf


def xgb(X, Y):
    """Train XGBoost on a default 75/25 split and print test accuracy.

    Used only by `test.py`.

    Parameters
    ----------
    X, Y : array-like
    """
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))


def votingClassifier(X, y, y_Orig, title):
    """Fit and evaluate a soft-voting ensemble of LogReg / MLP / SVC.

    Pipeline:
        1. 75/25 train/test split (random_state=1 for reproducibility).
        2. 5-fold cross-validated predictions + scores on the train fold.
        3. Refit on the train fold; predict probabilities on test.
        4. Render plots: prediction-vs-truth histogram, class-probability
           scatter, ROC, confusion matrix, per-classifier accuracy bars,
           and a 5-fold ROC averaged over folds.

    The three base estimators:
        * LogisticRegression(penalty='elasticnet', solver='saga',
                             l1_ratio=0.75)  -> sparse linear model.
        * MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs',
                        alpha=1e-5)           -> small dense net.
        * SVC(probability=True)               -> RBF kernel by default.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Binary classification target (0/1) used to fit and score.
    y_Orig : array-like, shape (n_samples,)
        Original (un-binarized) target — passed to `plots.predGold` so the
        prediction-distribution plot can show the underlying GOLD/emph
        spread, not just the binary labels.
    title : str
        Title used on every emitted figure.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Base classifiers + soft-voting ensemble.
    clf1 = LogisticRegression(random_state=1, penalty='elasticnet',
                              solver='saga', l1_ratio=0.75)
    clf2 = MLPClassifier(hidden_layer_sizes=(5, 2), solver='lbfgs',
                         alpha=1e-5, random_state=1)
    clf3 = SVC(probability=True)

    eclf = VotingClassifier(
        estimators=[('lr', clf1), ('mlp', clf2), ('svc', clf3)],
        voting='soft',
    )

    # 5-fold CV predictions / probabilities / accuracy on the *train* split.
    y_predCV = cross_val_predict(eclf, X_train, y_train, cv=5)
    y_pred_probCV = cross_val_predict(eclf, X_train, y_train, cv=5,
                                      method='predict_proba')
    scores = cross_val_score(eclf, X_train, y_train, cv=5)

    # Refit the ensemble on the full train fold and score on test.
    eclf1 = eclf.fit(X_train, y_train)
    probs = eclf1.predict_proba(X_test)
    c = eclf1.predict(X_test)

    # Prediction-distribution histogram (GOLD scale).
    plots.predGold(c, y_Orig, title)
    # compute_feature_importance(eclf1, [1, 1, 1])  # ad-hoc hook

    # Soft-vote probability scatter (Emph axis labels are inherited from
    # earlier emphysema runs; rename in-place if you re-target the script).
    plt.scatter(probs[:, 0], probs[:, 1], c=c)
    plt.rc('font', size=14)
    plt.xlabel('Emph 0-5%')
    plt.ylabel('Emph 5%+')
    plt.title(title)
    plt.show()

    # Single-fit ROC on the held-out test split.
    metrics.plot_roc_curve(eclf, X_test, y_test)
    plt.rc('font', size=14)
    plt.title(title)
    plt.show()

    # Per-classifier 5-fold CV accuracy on test (and on train) — feeds the
    # `plots.classProbs` bar chart further down.
    probas = []
    for clf, label in zip(
        [clf1, clf2, clf3, eclf],
        ['Logistic Regression SAGA ElasticNet', 'MLP Classifier', 'SVC', 'Ensemble'],
    ):
        scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=5)
        scores_train = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)

        print("Accuracy: %0.2f (+/- %0.2f) [%s]" %
              (scores.mean(), scores.std(), label))
        print("Accuracy Train: %0.2f (+/- %0.2f) [%s]" %
              (scores_train.mean(), scores_train.std(), label))
        probas.append([scores.mean() * 100, scores.std()])

    # Confusion matrix (normalised). NOTE: `plot_confusion_matrix` was
    # removed in scikit-learn 1.2 — pin sklearn < 1.2 (see requirements.txt).
    cf = confusion_matrix(y_test, c, labels=[0, 1])
    plot_confusion_matrix(eclf, X_test, y_test, normalize='all',
                          display_labels=['GOLD 0', 'GOLD 2-4'])
    plt.yticks(rotation=90, va='center')
    plt.rc('font', size=14)
    plt.title(title)

    # Bar chart of per-classifier accuracy + 5-fold averaged ROC.
    labels = ['Logistic Regression SAGA', 'MLP Classifier', 'SVC', 'Ensemble']
    plots.classProbs(np.array(probas), labels, title)
    plots.kROC(eclf, X_test, y_test, title)
    return


def compute_feature_importance(voting_clf, weights):
    """Pull per-estimator feature-importance signals from a VotingClassifier.

    The three estimators expose different attributes:
        * LogisticRegression -> `.coef_`
        * MLPClassifier      -> `.coefs_` (list, one per layer)
        * SVC                -> `.dual_coef_` (support-vector coefficients)

    The function gathers each into a dict, prints a top-N summary for the
    first estimator, and returns a weighted sum across estimators.

    Parameters
    ----------
    voting_clf : sklearn.ensemble.VotingClassifier
        A *fitted* voting classifier — `.estimators_` must be available.
    weights : sequence of float, length == number of estimators

    Returns
    -------
    list[float]
        Per-feature weighted importance score.
    """
    feature_importance = dict()
    c = 0
    for est in voting_clf.estimators_:
        if c == 0:
            feature_importance[str(est)] = np.array(est.coef_)
            importance = est.coef_
            top = {}
            for i, v in enumerate(importance[0]):
                top[i] = v
            topS = dict(sorted(top.items(), key=lambda item: item[1]))

            count = 0
            for i in topS:
                print(i, topS[i])
                if count > 10:
                    break
                count += 1

            # Bar plot of all per-feature coefficients for the first estimator.
            plt.bar([x for x in range(len(importance[0]))], importance[0])
            plt.show()
        if c == 1:
            feature_importance[str(est)] = np.array(est.coefs_)
        if c == 2:
            feature_importance[str(est)] = np.array(est.dual_coef_)
        c += 1

    fe_scores = [0] * len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score * weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores
