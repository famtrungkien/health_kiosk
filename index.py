import pickle
import pandas as pd
import sklearn.metrics as skm

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import classification_report
import numpy as np
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
import xgboost as xgb
import seaborn as sn

import matplotlib

matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

#labels = ['Influenza', 'Corona', 'Other']
labels = ['Class 0', 'Class 1', 'Class 2']

title = f"Load Training Model and Test Data"
title += '\n' + '-' * len(title)
print(title)

def to_numerical(labels):
    return np.argmax(labels, axis=1)

title = f"Show result of Model"
title += '\n' + '-' * len(title)
print(title)
data = pd.read_csv(f"data/TestData.csv")

model = xgb.XGBClassifier()
model = pickle.load(open(f"models/Model.pkl", 'rb'))


y_test = data['Class']
X_test = data.loc[:, data.columns != 'Class']
y_score = model.predict_proba(X_test)
y_score1 = to_numerical(y_score)

print(skm.classification_report(y_test, y_score1))

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# cm = confusion_matrix(y_test, y_score1, labels=["0", "1", "2"])
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot().figure_.savefig(f"images/confusion_matrix_xgboost.png")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(
    y_test,
    y_score1
)

title = f"# Show confusion matrix"
title += '\n' + '-' * len(title)
print(title)

classes = ["0", "1", "2"]
df_cfm = pd.DataFrame(cm, index = classes, columns = classes)

plt.rcParams.update({'font.size': 30})
plt.rc('font', size=25)          # controls default text sizes
plt.rc('axes', titlesize=25)     # fontsize of the axes title
plt.rc('axes', labelsize=25)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=25)    # fontsize of the tick labels
plt.rc('ytick', labelsize=25)    # fontsize of the tick labels
plt.rc('legend', fontsize=25)    # legend fontsize
plt.rc('figure', titlesize=25)  # fontsize of the figure title


plt.figure(figsize = (10,7))
cfm_plot = sn.heatmap(df_cfm, annot=True)
cfm_plot.figure.savefig(f"images/confusion_matrix_xgboost.png")
print(cm)

title = f"# Show ROC curve and AUC"
title += '\n' + '-' * len(title)
print(title)
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = skm.roc_curve((y_test == i).astype(int), y_score[:, i], )
    roc_auc[i] = skm.auc(fpr[i], tpr[i])

plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(8, 6))

for i in range(3):
    plt.plot(fpr[i], tpr[i], label=f'Class {labels[i]} (AUC = %0.3f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    #plt.title('ROC curve',fontsize = 25)
    plt.xlabel('False positive rate',fontsize = 25)
    plt.ylabel('True positive rate',fontsize = 25)
    plt.legend(loc='lower right',fontsize = 20)

plt.savefig(f"images/roc_curve_xgboost.png")
plt.show()
