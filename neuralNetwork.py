from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

import dataExploration
import plots
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.preprocessing import FunctionTransformer

def rFC(X, Y):
    clf = RandomForestClassifier(n_estimators=18)
    clf = clf.fit(X,Y)
    return clf

def xgb(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    model = XGBClassifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

def votingClassifier(X, y, y_Orig, title):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    #X_trainO, X_testO, y_trainO, y_testO = train_test_split(X,y_orig, random_state=1)


    clf1 = LogisticRegression(random_state=1, penalty='elasticnet', solver='saga', l1_ratio=0.75)
    clf2 = MLPClassifier(hidden_layer_sizes=(5,2), solver='lbfgs', alpha=1e-5, random_state=1)
    clf3 = SVC(probability=True)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('mlp', clf2), ('svc', clf3)], voting='soft')

    y_predCV = cross_val_predict(eclf, X_train, y_train, cv=5)
    y_pred_probCV = cross_val_predict(eclf, X_train, y_train, cv=5, method='predict_proba')
    scores = cross_val_score(eclf, X_train,y_train, cv=5)
    #print(scores)

    eclf1 = eclf.fit(X_train,y_train)
    probs = eclf1.predict_proba(X_test)
    c = eclf1.predict(X_test)

    #plots.predEmph(c, y_Orig, title)
    plots.predGold(c, y_Orig, title)

    #compute_feature_importance(eclf1, [1,1,1])

    plt.scatter(probs[:,0], probs[:,1],c=c)
    plt.rc('font', size=14)
    # (0, 1)
    # x, y
    #plt.xlabel('Smoking')
    #plt.xlabel('GOLD 0')
    plt.xlabel('Emph 0-5%')

    #plt.ylabel('Non Smoking')
    #plt.ylabel('GOLD 2-4')
    plt.ylabel('Emph 5%+')
    plt.title(title)
    plt.show()

    metrics.plot_roc_curve(eclf, X_test, y_test)
    plt.rc('font', size=14)
    plt.title(title)
    plt.show()


    probas = []
    for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression SAGA ElasticNet', 'MLP Classifier', 'SVC', 'Ensemble']):
        scores = cross_val_score(clf, X_test,y_test , scoring='accuracy', cv=5)
        scores_train = cross_val_score(clf, X_train,y_train , scoring='accuracy', cv=5)

        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        print("Accuracy Train: %0.2f (+/- %0.2f) [%s]" % (scores_train.mean(), scores_train.std(), label))
        probas.append([scores.mean()*100, scores.std()])


    cf = confusion_matrix(y_test, c, labels=[0,1])
    #plot_confusion_matrix(eclf, X_test, y_test, normalize='all', display_labels=['Emph 0-5%', 'Emph 5%+'])
    plot_confusion_matrix(eclf, X_test, y_test, normalize='all', display_labels=['GOLD 0', 'GOLD 2-4'])
    plt.yticks(rotation=90, va='center')
    plt.rc('font', size=14)
    plt.title(title)

    labels = ['Logistic Regression SAGA', 'MLP Classifier', 'SVC', 'Ensemble']
    plots.classProbs(np.array(probas), labels, title)

    plots.kROC(eclf, X_test, y_test, title)
    #plots.predEmph(c, y_testO, title)
    return


def compute_feature_importance(voting_clf, weights):
    """ Function to compute feature importance of Voting Classifier """
    feature_importance = dict()
    c=0
    for est in voting_clf.estimators_:
        if c == 0:
            feature_importance[str(est)] = np.array(est.coef_)
            importance = est.coef_
            top = {}
            for i, v in enumerate(importance[0]):
                top[i] = v
            topS = dict(sorted(top.items(), key=lambda  item:item[1]))

            count = 0
            for i in topS:

                print(i, topS[i])
                if count >10:
                    break
                count+=1

            # plot feature importance
            plt.bar([x for x in range(len(importance[0]))], importance[0])
            plt.show()
        if c == 1:
            feature_importance[str(est)] = np.array(est.coefs_)
        if c == 2:
            feature_importance[str(est)] = np.array(est.dual_coef_)
        c+=1
        #print(feature_importance[str(est)].shape)
        #print(feature_importance[str(est)])



    fe_scores = [0] * len(list(feature_importance.values())[0])
    for idx, imp_score in enumerate(feature_importance.values()):
        imp_score_with_weight = imp_score * weights[idx]
        fe_scores = list(np.add(fe_scores, list(imp_score_with_weight)))
    return fe_scores