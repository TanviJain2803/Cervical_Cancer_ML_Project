#import
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeRegressor
import graphviz
import pydotplus
#defining path and using pandas to read file
cancer_file_path = 'data/kag_risk_factors_cervical_cancer.csv'
cancer_data = pd.read_csv(cancer_file_path)
#check data by printing summary of csv file
print(cancer_data.describe())
#removing all rows with empty cells
cancer_data = cancer_data.dropna(axis=0)
#defining dependent variable
y = cancer_data.Biopsy
# defining independent variables
cancer_features = ['Age', 'Number of sexual partners', 'First sexual intercourse',
       'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)', 'IUD',
       'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis',
       'STDs:cervical condylomatosis', 'STDs:vaginal condylomatosis',
       'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
       'STDs:pelvic inflammatory disease', 'STDs:genital herpes',
       'STDs:molluscum contagiosum', 'STDs:AIDS', 'STDs:HIV',
       'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis',
       'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller',
       'Citology']
X = cancer_data[cancer_features]
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X, y, test_size = 0.3, random_state = 100)

def prediction(X_test, clf):
       # Predicton on test with giniIndex
       y_pred = clf.predict(X_test)
       print("Predicted values:")
       print(y_pred)
       return y_pred


def cal_accuracy(y_test, y_pred):
       print("Confusion Matrix: ",
             sk.metrics.confusion_matrix(y_test, y_pred))

       print("Accuracy : ",
             sk.metrics.accuracy_score(y_test, y_pred) * 100)

       print("Report : ",
             sk.metrics.classification_report(y_test, y_pred))

import Decision_tree
import Random_Forest
clf_gini = Decision_tree.train_using_gini(X_train, y_train)
Decision_tree.DecisionTreeclf(clf_gini, cancer_features)
y_pred_DecisionTree = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_DecisionTree)
regressor = Random_Forest.RandomForestRegressorCervical(X_train, y_train, X_test)
print(regressor)
Y_pred_RandoForest = prediction(X_test, regressor)
cal_accuracy(y_test, Y_pred_RandoForest)

'''
cm = confusion_matrix(val_y, val_predictions)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
'''