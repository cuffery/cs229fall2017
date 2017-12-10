import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


import logistic
import model_selection
import svm

import joined_data_pred

def main():
    #-----PREPARE DATA-----#
    label_, feature_ = joined_data_pred.importData()

    data_ = joined_data_pred.joinLabelFeature(label_, feature_)

    # remove useless data cols from data_
    data = joined_data_pred.cleanDataForSklearn(data_)

    #split into train set, dev set, test set
    all_train, test = train_test_split(data, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    after_set_1_all_train = all_train[all_train['after_set']==1]
    after_set_2_all_train = all_train[all_train['after_set']==2]

    #-----GRAPH FEATURE SELECTION-----#
    df_log_set1 = model_selection.getRFEMeanSTD(LogisticRegression(),after_set_1_all_train.drop(['label'], axis=1),after_set_1_all_train['label'],'Logistic Regression After Set 1')
    df_log_set2 = model_selection.getRFEMeanSTD(LogisticRegression(),after_set_2_all_train.drop(['label'], axis=1),after_set_2_all_train['label'],'Logistic Regression After Set 2')
    df_linearSVC_set1 = model_selection.getRFEMeanSTD(LinearSVC(),after_set_1_all_train.drop(['label'], axis=1),after_set_1_all_train['label'],'Linear SVC After Set 1')
    df_linearSVC_set2 = model_selection.getRFEMeanSTD(LinearSVC(),after_set_2_all_train.drop(['label'], axis=1),after_set_2_all_train['label'],'Linear SVC After Set 2')

    fig, axs = plt.subplots(ncols=2, sharex=True)
    ax = axs[1]
    ax.errorbar(df_log_set2['Feature #'], df_log_set2['Mean Score'], yerr=df_log_set2['St.Dev'],label = "Logistic Regression",fmt = 'o')
    ax.errorbar(df_linearSVC_set2['Feature #'], df_linearSVC_set2['Mean Score'], yerr=df_linearSVC_set2['St.Dev'],label = "Linear SVC",fmt = 'o')
    ax.set_ylim([0.5,1])
    ax.legend(loc = 1)
    ax.set_title('After set 2 data')

    ax = axs[0]
    ax.errorbar(df_log_set1['Feature #'], df_log_set1['Mean Score'], yerr=df_log_set1['St.Dev'],label = "Logistic Regression",fmt = 'o')
    ax.errorbar(df_linearSVC_set1['Feature #'], df_linearSVC_set1['Mean Score'], yerr=df_linearSVC_set1['St.Dev'],label = "Linear SVC",fmt = 'o')
    ax.legend(loc = 1)
    ax.set_ylim([0.5,1])

    ax.set_title('After set 1 data')

    plt.savefig('Error Comparison with Feature Size for Logistic and SVC.png')
    

    return

if __name__ == '__main__':
	main()