import sys
import pickle
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import classification_report
import numpy as np
from yellowbrick.classifier import ROCAUC

labels = ['Influenza', 'Covid-19', 'Other']



obj = pd.read_pickle(f'models/Model.pkl')
print(obj)

title = f"Validation Covid-19 Flu and Other"
title += '\n' + '-' * len(title)
print(title)

data = pd.read_csv(f"data/Data_Test.csv")
model = pickle.load(open(f"models/Model.pkl", 'rb'))
print(skm.classification_report(data['Class'], model.predict(data.drop('Class', axis=1)), target_names=labels))





y_test = data['Class']
X_test = data.loc[:, data.columns != 'Class']

y_score = model.predict_proba(X_test)



from yellowbrick.classifier import ConfusionMatrix


cm = ConfusionMatrix(model)
cm.score(X_test, y_test)

cm.ax.tick_params(labelsize=22)  # change size of tick labels
cm.ax.title.set_fontsize(30)  # change size of title

for xtick,ytick in zip(cm.ax.xaxis.get_major_ticks(),cm.ax.yaxis.get_major_ticks()):
    xtick.label.set_fontsize(30) 
    ytick.label.set_fontsize(30)


for label in cm.ax.texts:
    label.set_size(24)
cm.show()


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = skm.roc_curve((y_test == i).astype(int), y_score[:, i], )
    roc_auc[i] = skm.auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))

plt.rcParams.update({'font.size': 22})

for i in range(3):
    
    plt.plot(fpr[i], tpr[i], label=f'Class {labels[i]} (AUC = %0.3f)' % roc_auc[i])
  
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC curve',fontsize = 22)
    plt.xlabel('False positive rate',fontsize = 20)
    plt.ylabel('True positive rate',fontsize = 20)
    plt.legend(loc='best')
plt.show()

