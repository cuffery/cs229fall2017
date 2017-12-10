
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


'''
This file contains all model selection related functions, including:

Inputs are:
- Training data + dev data
- Model .fit return
'''


def getCrossValScore(model, feature, label,fold):
    #takes a model, training data, and training label, and output the scores for all kfold
    return cross_val_score(model, feature, label, cv=fold)

def checkC(model,feature, label):
    #test for different c value in SVM, LogisticRegression
    for c in np.arange(0.1,5,0.1):
        if model == "SVM":
            scores = getCrossValScore(SVC(C = c),feature,label,10)
            print("For c =", c, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        if model == "LogisticRegression":
            scores = getCrossValScore(LogisticRegression(C = c),feature,label,10)
            print("For c =", c, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        else:
            raise Exception("This input for model is not supported, input model as a string")
    return

def checkKSVM(feature, label):
    #test for different kernel value in SVM, not very useful, other than rbf others can't run
    kernel = ['linear','poly', 'rbf', 'sigmoid']

    for k in kernel:
        scores = getCrossValScore(SVC(kernel=k),feature,label,3)
        print("For kernel =", k, "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        #print("For c =", c, "scores",scores)

def getRFE(model,num):
    #returns the rfe and the dataframe that contains feature ranking and feature names
    #model needs to be instantiated
    #num is the number of features to select
    return RFE(model,num)

def getRFERanking(rfe,feature,label):
    fit = rfe.fit(feature, label)
    feature_rank = pd.DataFrame({'label':list(feature),'Ranking':fit.ranking_,'Selected':fit.support_})
    return feature_rank.sort_values(by = ['Ranking'])

def getRFEMeanSTD(model,feature,label,desc):
    score_mean = []
    feature_num = []
    score_std = []

    for i in range(1,len(list(feature))):
        rfe = getRFE(model,i)
        rfe = rfe.fit(feature,label)
        scores = getCrossValScore(rfe,feature,label,5)
        print(desc,":", "For",i,"features selected, accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        feature_num.append(i)
        score_mean.append(scores.mean())
        score_std.append(scores.std())
    return pd.DataFrame({'Feature #':feature_num,'Mean Score':score_mean,'St.Dev':score_std})



def main():
    

    return


if __name__ == '__main__':
	main()