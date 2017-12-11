'''
  model_evaluation.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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


def plot_learning_curve(train_sizes, train_scores, test_scores, filename, debug):
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.figure()
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score" )
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Test score")
    plt.ylim([0.5,1])
    plt.xlabel('# of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")

    plt.title(filename[0:-4])
    plt.savefig(filename)

    return

def get_learning_curve_plots(diff):    
    #read data with diff model
    label_, feature_ = joined_data_pred.importData(diff)
    data_ = joined_data_pred.joinLabelFeature(label_, feature_)

    # remove useless data cols from data_
    data = joined_data_pred.cleanDataForSklearn(data_)

    all_train, test = train_test_split(data, test_size=0.20, random_state=666)

    after_set_1_all_train = all_train[all_train['after_set']==1]
    after_set_2_all_train = all_train[all_train['after_set']==2]

    y = after_set_1_all_train['label']
    
    drop_col = ['label','after_set']
    
    X = after_set_1_all_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')

    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(X,y)

    train_sizes_lin_svc, train_scores_lin_svc, valid_scores_lin_svc = learning_curve(LinearSVC(), rfe.transform(X), y, train_sizes=np.linspace(.1, 1.0, 5), cv=5)

    print("train_sizes_lin_svc\n", train_sizes_lin_svc)
    print("train_scores_lin_svc\n", train_scores_lin_svc)
    print("valid_scores_lin_svc\n", valid_scores_lin_svc)

    plot_learning_curve(train_sizes_lin_svc, train_scores_lin_svc, valid_scores_lin_svc, 'LinearSVC_learning_curve.png', 1)

    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(X,y)

    train_sizes_svm, train_scores_svm, valid_scores_svm = learning_curve(SVC(), rfe.transform(X), y, train_sizes=np.linspace(.1, 1.0, 5), cv=5)

    print("train_sizes_svc\n", train_sizes_svm)
    print("train_scores_svc\n", train_scores_svm)
    print("valid_scores_svc\n", valid_scores_svm)

    plot_learning_curve(train_sizes_svm, train_scores_svm, valid_scores_svm, 'SVM_learning_curve.png', 2)

    train_sizes_lr, train_scores_lr, valid_scores_lr = learning_curve(LogisticRegression(), X, y,  train_sizes=np.linspace(.1, 1.0, 5), cv=5)

    print("train_sizes_lr\n", train_sizes_lr)
    print("train_scores_lr\n", train_scores_lr)
    print("valid_scores_lr\n", valid_scores_lr)
    plot_learning_curve(train_sizes_lr, train_scores_lr, valid_scores_lr, 'LR_learning_curve.png', 3)
    return

def getConfusionMatrix(diff):
    #----GET CONFUSION MATRIX FOR BEST MODEL----#
    #read data with diff model
    label_, feature_ = joined_data_pred.importData(diff)
    data_ = joined_data_pred.joinLabelFeature(label_, feature_)
    drop_col = ['label','after_set']

    # remove useless data cols from data_
    data = joined_data_pred.cleanDataForSklearn(data_)

    all_train, test = train_test_split(data, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    after_set_1_all_train = all_train[all_train['after_set']==1]
    after_set_2_all_train = all_train[all_train['after_set']==2]

    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct','ttl_pts_won']

    after_set_1_train = train[train['after_set']==1]
    after_set_2_train = train[train['after_set']==2]

    after_set_1_dev = dev[dev['after_set']==1]
    after_set_2_dev = dev[dev['after_set']==2]

    #historical confusion matrix
    svc = LinearSVC()
    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(train[hist],train['label'])
    svc.fit(rfe.transform(train[hist]),train['label'])
    pred = svc.predict(rfe.transform(dev[hist]))
    
    tn, fp, fn, tp = confusion_matrix(dev['label'],pred).ravel()
    print('Historical')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current only after set 1 confusion matrix
    svc = LinearSVC()
    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(after_set_1_train[curr],after_set_1_train['label'])
    svc.fit(rfe.transform(after_set_1_dev[curr]),after_set_1_dev['label'])
    pred = svc.predict(rfe.transform(after_set_1_dev[curr]))
    
    tn, fp, fn, tp = confusion_matrix(after_set_1_dev['label'],pred).ravel()
    print('Current only after set 1')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current + historical only after set 1 confusion matrix
    svc = LinearSVC()
    X = after_set_1_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    y = after_set_1_train['label']
    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(X,y)
    svc.fit(rfe.transform(X),y)
    pred = svc.predict(rfe.transform(after_set_1_dev.drop(drop_col,axis = 1).dropna(axis=0, how='any')))
    
    tn, fp, fn, tp = confusion_matrix(after_set_1_dev['label'],pred).ravel()
    print('Current + historical after set 1')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current + historical only after set 2 confusion matrix
    svc = LinearSVC()
    X = after_set_2_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    y = after_set_2_train['label']
    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe = rfe.fit(X,y)
    svc.fit(rfe.transform(X),y)
    pred = svc.predict(rfe.transform(after_set_2_dev.drop(drop_col,axis = 1).dropna(axis=0, how='any')))
    
    tn, fp, fn, tp = confusion_matrix(after_set_2_dev['label'],pred).ravel()
    print('Current + historical after set 2')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

def main():
    get_learning_curve_plots(diff = True)
    getConfusionMatrix(diff = True)

    return


if __name__ == '__main__':
    main()