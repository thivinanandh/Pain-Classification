# %% 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc
import sys
from tqdm import tqdm
#import random forest classifier
from sklearn.ensemble import RandomForestClassifier
#import svc
from sklearn.svm import SVC
#import logistic regression
from sklearn.linear_model import LogisticRegression
#import KNN
from sklearn.neighbors import KNeighborsClassifier
#import gaussianNB
from sklearn.naive_bayes import GaussianNB
#import decision tree
from sklearn.tree import DecisionTreeClassifier
# import mlp
from sklearn.neural_network import MLPClassifier


from sklearn.model_selection import GridSearchCV
##import cross validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold


import pickle


# %%
## function to read file and return list of mean values of each row
def read_file(filename):
    rowArray = []
    mainDict = {}

    condDict = {
        "BP Dia_mmHg": "dia",
        "LA Systolic BP_mmHg": "sys",
        "EDA_microsiemens": "eda",
        "Respiration Rate_BPM": "res",
    }

    with open(filename) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        linesArray = [line.split(',') for line in lines]
        for line in tqdm(linesArray):
            name= line[0].lower()
            cond = line[1]
            classVal = line[2]
            if(classVal == "No Pain"):
                classVal = 0
            else:
                classVal = 1
            cond = condDict[cond]
            
            
            line = [float(i) for i in line[3:]]
            [min,max,mean,variance] = [np.min(line),np.max(line),np.mean(line),np.var(line)]
            rowArray.append([name,cond,classVal,min,max,mean,variance])
    return np.array(rowArray)
# %%

if __name__ == "__main__":
    # check no of arguments
    # check number of arguments
    if(len(sys.argv) != 2):
        print("Usage: python3 painClassification.py <datatype>")
        exit(1)
    
    # check if datatype is valid
    if(sys.argv[1] not in ["all","sys","dia","eda","res"]):
        print("Invalid datatype")
        print("Valid datatypes are : all, sys, dia, eda, res")
        exit(1)


    print("Reading file from csv")
    MAIN_ROW_ARRAY = read_file("Project2Data.csv")
    

    # %% 

    # Create Dataframe
    df = pd.DataFrame(MAIN_ROW_ARRAY,columns=["name","cond","class","min","max","mean","variance"])
    # pivot on cond column
    df = df.pivot_table(index=["name","class"],columns="cond",values=["min","max","mean","variance"]).reset_index().reset_index()

    df.columns = df.columns.map('_'.join)

    # %%

    # Get column names for given condition
    def get_col_names(cond):
        if(cond != "all"):
            return [f"{i}_{cond}" for i in ["min","max","mean","variance"]]
        else:
            return [f"{i}_{cond}" for i in ["min","max","mean","variance"] for cond in ["sys","dia","eda","res"]]

    # %%

    # %%

    ## Generate a classification models using pipeline
    # Get X and Y for given condition

    givenCondition = sys.argv[1]
    x,y = df[get_col_names(givenCondition)], df["class_"]
    X = x.values
    Y = y.values
    Y = Y.astype('int')
    pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier())])






    # %%

    # givenCondition = "dia"
    print("Selected DataType : ",givenCondition)
    x,y = df[get_col_names(givenCondition)], df["class_"]
    X = x.values
    Y = y.values
    Y = Y.astype('int')

    ## get  split of kfold
    skf = KFold(n_splits=10)

    pipeline = Pipeline([('scaler', StandardScaler()), ('model', RandomForestClassifier(
                                                        criterion =  "entropy",
                                                        max_depth =  5,
                                                        max_features =  5,
                                                        n_estimators =  10)
                                                        )])

    CF_ARRAY = []
    ACCURACY_ARRAY= []
    PRECISION_ARRAY = []
    RECALL_ARRAY = []
    F1_ARRAY = []

    for i, (train_index, test_index) in enumerate(skf.split(X,Y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        pipeline.fit(X_train,y_train)
        y_pred = pipeline.predict(X_test)
        ACCURACY_ARRAY.append(accuracy_score(y_test,y_pred))
        PRECISION_ARRAY.append(precision_score(y_test,y_pred))
        RECALL_ARRAY.append(recall_score(y_test,y_pred))
        CF_ARRAY.append(confusion_matrix(y_test,y_pred))
        F1_ARRAY.append(f1_score(y_test,y_pred))
        # print("Confusion Matrix: ",confusion_matrix(y_test,y_pred))
        # print("Accuracy: ",accuracy_score(y_test,y_pred))
        # print("Precision: ",precision_score(y_test,y_pred))
        # print("Recall: ",recall_score(y_test,y_pred))
        # print("F1 Score: ",f1_score(y_test,y_pred))

    print("Accuracy   : ",np.mean(ACCURACY_ARRAY))
    print("Precision  : ",np.mean(PRECISION_ARRAY))
    print("Recall     : ",np.mean(RECALL_ARRAY))
    print("F1 Score   : ",np.mean(F1_ARRAY))
    print("Confusion Matrix: \n",np.mean(CF_ARRAY,axis=0))



                
# %%

## ------------------ ADDITIONAL CODES ------------------ ##
## Gridsearch cv for multiple classifiers
# param_grid = [
#     {'model': [RandomForestClassifier()],
#      'model__n_estimators': [10, 100, 1000],
#      'model__max_features': [2, 4,  8],
#      'model__max_depth': [2, 4, 6, 10],
#      'model__bootstrap': [True, False],
#      'model__criterion': ['gini', 'entropy']
#      },
#     {'model': [SVC()],
#      'model__kernel': ['linear', 'rbf'],
#      'model__C': [0.1, 1, 10,1000],
#      'model__gamma': [1, 0.001, 0.0001],
#      'model__degree': [1, 2, 3 ]
#      },

#     {'model': [KNeighborsClassifier()],
#      'model__n_neighbors': [1, 2, 3, 4, 5, 6, 7],
#      'model__weights': ['uniform', 'distance'],
#      'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
#     #  'model__leaf_size': [10, 20, 30, 40, 50, 60]
#      }

# ]

# grid = GridSearchCV(pipeline, param_grid=param_grid, cv=10, n_jobs=-1, verbose=10)
# grid.fit(X, Y)
# print(grid.best_params_)
# print(grid.best_score_)
# print(grid.best_estimator_)
# print(grid.cv_results_)
# print(grid.scorer_)
# print(grid.refit)
# disp = ConfusionMatrixDisplay(confusion_matrix=k,display_labels=["No Pain","Pain"])
# disp.plot()

 # %%

## BarChart
# dfOrig = pd.DataFrame(MAIN_ROW_ARRAY,columns=["name","cond","class","min","max","mean","variance"])
# dfOrig["class"] = dfOrig["class"].astype('int')
# dfOrig["min"] = dfOrig["min"].astype('float')
# dfOrig["max"] = dfOrig["max"].astype('float')
# dfOrig["mean"] = dfOrig["mean"].astype('float')
# dfOrig["variance"] = dfOrig["variance"].astype('float')
# dfOrig["cond"] = dfOrig["cond"].astype('str')
# ax = sns.boxplot(data=dfOrig, x="class", y="min",hue="cond")
# ax.set_title("Minimum - Box plot")
# plt.savefig("min_boxplot.png")
# plt.figure()
# ax = sns.boxplot(data=dfOrig, x="class", y="max",hue="cond")
# ax.set_title("Maximum - Box plot")
# plt.savefig("max_boxplot.png")
# plt.figure()
# ax = sns.boxplot(data=dfOrig, x="class", y="mean",hue="cond")
# ax.set_title("Mean - Box plot")
# plt.savefig("mean_boxplot.png")
# plt.figure()
# ax = sns.boxplot(data=dfOrig, x="class", y="variance",hue="cond")
# ax.set_title("Variance - Box plot")
# plt.savefig("variance_boxplot.png")


# # %%
# dfOrig = pd.DataFrame(MAIN_ROW_ARRAY,columns=["name","cond","class","min","max","mean","variance"])

# dfOrig = dfOrig.loc[dfOrig["cond"] == "sys"]
# # melt dataframe
# dfOrig = pd.melt(dfOrig, id_vars=["name","cond","class"], value_vars=["min","max","mean","variance"], var_name="stat", value_name="value")
# dfOrig["class"] = dfOrig["class"].astype('int')
# dfOrig["value"] = dfOrig["value"].astype('float')
# dfOrig["stat"] = dfOrig["stat"].astype('str')


# plt.figure()
# ax = sns.boxplot(data=dfOrig, x="class", y="value",hue="stat")
# ax.set_title("Sys - Box plot")
# ax.set_ylim(0, 600)
# plt.savefig("sys_boxplot.png")

# # %% 
# dfOrig = pd.DataFrame(MAIN_ROW_ARRAY,columns=["name","cond","class","min","max","mean","variance"])
# dfOrig["variance"] = dfOrig["variance"].astype('float')
# ## line plot of 
# ax = dfOrig.loc[dfOrig["cond"] == "sys"]['variance'].plot(kind="line",label="sys")
# ax = dfOrig.loc[dfOrig["cond"] == "dia"]['variance'].plot(kind="line",label="dia")
# ax = dfOrig.loc[dfOrig["cond"] == "eda"]['variance'].plot(kind="line",label="eda")
# ax = dfOrig.loc[dfOrig["cond"] == "res"]['variance'].plot(kind="line",label="res")
# plt.legend()
# plt.savefig("Variance_lineplot.png")

# plt.figure()
# df2  = dfOrig.loc[dfOrig["cond"] == "sys"]

# ax = df2.loc[df2["class"] == '0']['variance'].plot(kind="line",label="sys - no pain")
# ax = df2.loc[df2["class"] == '1']['variance'].plot(kind="line",label="sys - pain")
# plt.legend()
# plt.savefig("Variance_lineplot_sys.png")


# %%
