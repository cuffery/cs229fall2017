
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE


'''
This file contains all model selection related functions, including:
- forward error analysis for a given model
- ablative error analysis for a given model
- k-cross validation on a given model
- k-cross validation comparison on a set of models

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

def getRFE(model,feature, label,num):
    #returns the rfe and the dataframe that contains feature ranking and feature names
    #model needs to be instantiated
    #num is the number of features to select
    rfe = RFE(model,num)
    fit = rfe.fit(feature, label)
    feature_rank = pd.DataFrame({'label':list(feature),'Ranking':fit.ranking_,'Selected':fit.support_})
    
    return rfe, feature_rank



def main():
    

    return


if __name__ == '__main__':
	main()