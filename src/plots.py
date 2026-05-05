"""
Matplotlib plotting helpers used by the voting-classifier evaluation routine.

Each function draws a single figure and calls `plt.show()` directly, so the
script blocks until the window is dismissed (Cowork / IDE plot panes will
display them inline).

    classProbs   — grouped bar chart: per-classifier mean accuracy on test
                   (correct vs incorrect), with the ensemble's bars styled
                   in blue to stand out from the three base classifiers.
    kROC         — 5-fold stratified ROC: per-fold curves plus mean ROC
                   with ±1 std band.
    predEmphBi   — prediction-distribution histogram for binary emphysema.
    predEmph     — prediction-distribution histogram against the *continuous*
                   pctEmph values (so you can see how the binary prediction
                   maps back onto the original target).
    predGold     — prediction-distribution histogram for GOLD stage (uses
                   integer bins 0..4).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc


def classProbs(probas, labels, title):
    """Grouped bar chart of per-classifier accuracy on the test split.

    The first three classifier rows are drawn in green (correct/incorrect);
    the final row (the voting ensemble) is drawn in blue so it visually
    pops out from the base classifiers. A vertical dashed line separates
    the base estimators from the ensemble.

    Parameters
    ----------
    probas : numpy.ndarray, shape (4, 2)
        Rows: [LogReg, MLP, SVC, Ensemble]. Columns: [mean_acc * 100, std].
        Only the first column (mean accuracy) is plotted.
    labels : list[str]
        Reserved for future use (current x-tick labels are hard-coded
        below to keep figure styling consistent across runs).
    title : str
        Figure title.
    """
    fig, ax = plt.subplots()
    width = 0.35
    ind = np.arange(4)

    probas0 = probas[:, 0]               # mean accuracy (%)
    probas1 = 100 - probas[:, 0]         # complement = "incorrect" %

    # Green pair = base classifiers (LogReg, MLP, SVC)
    p1 = ax.bar(ind, np.hstack(([probas0[:-1], [0]])), width,
                color="green", edgecolor="k")
    p2 = ax.bar(ind + width, np.hstack(([probas1[:-1], [0]])), width,
                color="lightgreen", edgecolor="k")

    # Blue pair = voting ensemble (only the last column is non-zero)
    p3 = ax.bar(ind, [0, 0, 0, probas0[-1]], width,
                color="blue", edgecolor="k")
    p4 = ax.bar(ind + width, [0, 0, 0, probas1[-1]], width,
                color="steelblue", edgecolor="k")

    # Visual separator between base classifiers and the ensemble.
    plt.axvline(2.8, color="k", linestyle="dashed")
    ax.set_xticks(ind + width)
    ax.set_xticklabels(
        ["Logistic Regression\n", "MP CLassifier\n", "SVC\n", "Voting Classifier\n"],
        rotation=40, ha="right",
    )
    plt.ylim([0, 100])
    plt.title(title)
    plt.ylabel('Percent Predicted')
    plt.legend([p1[0], p2[0]], ["Correct", "Incorrect"], loc="upper left")
    plt.tight_layout()
    plt.show()
    return


def kROC(classifier, X, y, title):
    """Stratified-5-fold ROC: per-fold curves + mean ROC with ±1 std band.

    Refits `classifier` from scratch on each fold, plots the per-fold ROC
    semi-transparently, then overlays the mean ROC across folds and a grey
    band representing one standard deviation of the true-positive rate at
    each FPR.

    Parameters
    ----------
    classifier : estimator
        An unfitted scikit-learn estimator implementing `.fit` and
        `predict_proba`. It is refit on each fold.
    X : numpy.ndarray, shape (n_samples, n_features)
    y : numpy.ndarray, shape (n_samples,)
    title : str
        Figure title.
    """
    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier, X[test], y[test],
            name="ROC fold {}".format(i),
            alpha=0.3, lw=1, ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    # Chance line.
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r",
            label="Chance", alpha=0.8)

    # Mean ROC across folds.
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color="b",
            label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
            lw=2, alpha=0.8)

    # ±1 std band.
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                    color="grey", alpha=0.2,
                    label=r"$\pm$ 1 std. dev.")

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=title)
    ax.legend(loc="lower right", prop={'size': 12})
    plt.show()


def predEmphBi(y_pred, y_Orig, title):
    """Prediction-distribution histogram for *binary* emphysema labels.

    Buckets the original (binary) emphysema labels by the model's predicted
    class — useful as a quick sanity check that the predicted classes span
    the original distribution.
    """
    bars = {1: [], 0: []}
    for y in range(len(y_pred)):
        bars[y_pred[y]].append(y_Orig[y])

    plt.hist([bars[0], bars[1]],
             color=['skyblue', 'red'],
             label=['Emph 0-5', 'Emph5+'])
    plt.legend()
    plt.title(title)
    plt.show()


def predEmph(y_pred, y_Orig, title):
    """Prediction-distribution histogram against *continuous* pctEmph.

    For each predicted class, plot the distribution of the underlying
    pctEmph values that fell into that bin. Lets you see how cleanly the
    binary classifier separates the original continuous target.
    """
    bars = {1: [], 0: []}
    for y in range(len(y_pred)):
        bars[y_pred[y]].append(y_Orig[y])
        print(y_pred[y], y_Orig[y])

    plt.hist([bars[0], bars[1]],
             color=['skyblue', 'red'],
             label=['Predicted emph 0-5', 'emph 5+'])
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.xlabel('Percent Emphysema')
    plt.ylabel('Number of Patients')
    plt.legend()
    plt.title('Prediction Distribution ' + title)
    plt.show()


def predGold(y_pred, y_Orig, title):
    """Prediction-distribution histogram against original GOLD stage (0..4).

    Same idea as `predEmph` but bucketed by integer GOLD stage rather than
    continuous emphysema percent.
    """
    bars = {1: [], 0: []}
    for y in range(len(y_pred)):
        bars[y_pred[y]].append(y_Orig[y])
        print(y_pred[y], y_Orig[y])

    plt.hist([bars[0], bars[1]],
             color=['skyblue', 'red'],
             label=['Gold 0', 'Gold 2-4'])
    plt.xticks([0, 1, 2, 3, 4])
    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
    plt.xlabel('Gold Stage')
    plt.ylabel('Number of Patients')
    plt.legend()
    plt.title('Prediction Distribution ' + title)
    plt.show()
