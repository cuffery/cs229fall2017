'''
  model_evaluation.py
'''
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import logistic
import svm
import joined_data_pred
import model_selection


def plot_learning_curve():

    return


def main():
    # read data
    label_, feature_ = joined_data_pred.importData(False)

    data_ = joined_data_pred.joinLabelFeature(label_, feature_)

    # remove useless data cols from data_
    data = joined_data_pred.cleanDataForSklearn(data_)

    print(list(data))

    y = data['label']

    #drop_col = ['label','surface_hard', 'surface_grass', 'surface_clay']
    drop_col = ['label']

    X = data.drop(drop_col,axis = 1).dropna(axis=0, how='any')

    # [TODO: remove extra col-dropping] this is trying to speed things up a bit temporarily
    #temp_drop = ['ace_diff', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct', 'ttl_pts_won']
    #X = data.drop(temp_drop,axis = 1).dropna(axis=0, how='any')

    rfe = model_selection.getRFE(LinearSVC(),20)
    rfe = rfe.fit(X,y)

    train_sizes_lin_svc, train_scores_lin_svc, valid_scores_lin_svc = learning_curve(LinearSVC(), rfe.transform(X), y, train_sizes=[1000, 2000, 4000], cv=5)

    print("train_sizes_lin_svc\n", train_sizes_lin_svc)
    print("train_scores_lin_svc\n", train_scores_lin_svc)
    print("valid_scores_lin_svc\n", valid_scores_lin_svc)

    rfe = model_selection.getRFE(LinearSVC(),20)
    rfe = rfe.fit(X,y)

    train_sizes_svm, train_scores_svm, valid_scores_svm = learning_curve(SVC(), rfe.transform(X), y, train_sizes=[1000, 2000, 4000], cv=5)

    print("train_sizes_svc\n", train_sizes_svm)
    print("train_scores_svc\n", train_scores_svm)
    print("valid_scores_svc\n", valid_scores_svm)

    #plot_learning_curve()
    print(list(X))

    train_sizes_lr, train_scores_lr, valid_scores_lr = learning_curve(LogisticRegression(), X, y, train_sizes=[2000, 3000, 4000], cv=5)

    print("train_sizes_lr\n", train_sizes_lr)
    print("train_scores_lr\n", train_scores_lr)
    print("valid_scores_lr\n", valid_scores_lr)

    #plot_learning_curve()


    return


if __name__ == '__main__':
    main()