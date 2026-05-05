import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay, auc

def classProbs(probas, labels, title):
   fig, ax =plt.subplots()
   width = 0.35
   ind = np.arange(4)

   probas0 = probas[:,0]
   probas1 = 100 - probas[:,0]

   # bars for classifier 1-3
   p1 = ax.bar(ind, np.hstack(([probas0[:-1], [0]])), width, color="green", edgecolor="k")
   p2 = ax.bar(
      ind + width,
      np.hstack(([probas1[:-1], [0]])),
      width,
      color="lightgreen",
      edgecolor="k",
   )

   # bars for VotingClassifier
   p3 = ax.bar(ind, [0, 0, 0, probas0[-1]], width, color="blue", edgecolor="k")
   p4 = ax.bar(
      ind + width, [0, 0, 0, probas1[-1]], width, color="steelblue", edgecolor="k"
   )

   # plot annotations
   plt.axvline(2.8, color="k", linestyle="dashed")
   ax.set_xticks(ind + width)
   ax.set_xticklabels(
      [
         "Logistic Regression\n",
         "MP CLassifier\n",
         "SVC\n",
         "Voting Classifier\n",
      ],
      rotation=40,
      ha="right",
   )
   plt.ylim([0, 100])
   #plt.title("Average Class probabilities for the different classifiers")
   plt.title(title)
   plt.ylabel('Percent Predicted')
   plt.legend([p1[0], p2[0]], ["Correct", "Incorrect"], loc="upper left")
   plt.tight_layout()
   plt.show()

   return


def kROC(classifier, X, y, title):
   cv = StratifiedKFold(n_splits=5)

   tprs = []
   aucs = []
   mean_fpr = np.linspace(0, 1, 100)

   fig, ax = plt.subplots()
   for i, (train, test) in enumerate(cv.split(X, y)):
      classifier.fit(X[train], y[train])
      viz = RocCurveDisplay.from_estimator(
         classifier,
         X[test],
         y[test],
         name="ROC fold {}".format(i),
         alpha=0.3,
         lw=1,
         ax=ax,
      )
      interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
      interp_tpr[0] = 0.0
      tprs.append(interp_tpr)
      aucs.append(viz.roc_auc)

   ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

   mean_tpr = np.mean(tprs, axis=0)
   mean_tpr[-1] = 1.0
   mean_auc = auc(mean_fpr, mean_tpr)
   std_auc = np.std(aucs)
   ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
      lw=2,
      alpha=0.8,
   )

   std_tpr = np.std(tprs, axis=0)
   tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
   tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
   ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
   )

   ax.set(
      xlim=[-0.05, 1.05],
      ylim=[-0.05, 1.05],
      # title="Receiver operating characteristic example",
      title=title,
   )
   ax.legend(loc="lower right", prop={'size' : 12})
   plt.show()


def predEmphBi(y_pred, y_Orig, title):
   bars = {1:[], 0:[]}

   for y in range(len(y_pred)):
      bars[y_pred[y]].append(y_Orig[y])

   plt.hist([bars[0],bars[1]] ,color=['skyblue','red'], label=['Emph 0-5', 'Emph5+'])
   plt.legend()
   plt.title(title)
   plt.show()

def predEmph(y_pred, y_Orig, title):
   bars = {1:[], 0:[]}

   for y in range(len(y_pred)):
       bars[y_pred[y]].append(y_Orig[y])
       print(y_pred[y], y_Orig[y])

   plt.hist([bars[0],bars[1]] ,color=['skyblue','red'], label=['Predicted emph 0-5', 'emph 5+'])
   plt.xticks([0,5, 10,15,20,25,30,35,40])
   plt.yticks([0,10,20,30,40,50,60,70,80,90,100])
   plt.xlabel('Percent Emphysema')
   plt.ylabel('Number of Patients')
   plt.legend()
   plt.title('Prediction Distribution '+title)
   plt.show()


def predGold(y_pred, y_Orig, title):
   bars = {1:[], 0:[]}

   for y in range(len(y_pred)):
      bars[y_pred[y]].append(y_Orig[y])
      print(y_pred[y], y_Orig[y])

   plt.hist([bars[0],bars[1]] ,color=['skyblue','red'], label=['Gold 0', 'Gold 2-4'])
   plt.xticks([0,1,2,3,4])
   plt.yticks([0,10,20,30,40,50,60,70,80])
   plt.xlabel('Gold Stage')
   plt.ylabel('Number of Patients')
   plt.legend()
   plt.title('Prediction Distribution '+title)
   plt.show()
