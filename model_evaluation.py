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

def get_learning_curve_plots(data_split,num_features):    
    after_set_1_all_train = data_split['after_set_1_all_train']
    after_set_2_all_train = data_split['after_set_2_all_train']

    all_train = data_split['all_train']
    test = data_split['test']

    drop_col = ['label']
    y = after_set_1_all_train['label']
    X = after_set_1_all_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')

    rfe = model_selection.getRFE(LinearSVC(),num_features)
    rfe = rfe.fit(X,y)

    train_sizes_lin_svc, train_scores_lin_svc, valid_scores_lin_svc = learning_curve(LinearSVC(), rfe.transform(X), y, train_sizes=np.linspace(.1, 1.0, 5), cv=5)

    print("train_sizes_lin_svc\n", train_sizes_lin_svc)
    print("train_scores_lin_svc\n", train_scores_lin_svc)
    print("valid_scores_lin_svc\n", valid_scores_lin_svc)

    plot_learning_curve(train_sizes_lin_svc, train_scores_lin_svc, valid_scores_lin_svc, 'LinearSVC_learning_curve.png', 1)

    rfe = model_selection.getRFE(LinearSVC(),num_features)
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

def getConfusionMatrix(data_split,num_features, model):
    after_set_1_train= data_split['after_set_1_train']
    after_set_2_train = data_split['after_set_2_train']
    after_set_1_all_train = data_split['after_set_1_all_train']
    after_set_2_all_train = data_split['after_set_2_all_train']
    after_set_1_dev = data_split['after_set_1_dev']
    after_set_2_dev = data_split['after_set_2_dev']

    all_train = data_split['all_train']
    test = data_split['test']
 
    train = data_split['train']
    dev = data_split['dev']
    drop_col = ['label']

    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct','ttl_pts_won']

    #historical confusion matrix
    svc = model
    rfe = model_selection.getRFE(model,num_features)
    rfe = rfe.fit(train[hist],train['label'])
    svc.fit(rfe.transform(train[hist]),train['label'])
    pred = svc.predict(rfe.transform(dev[hist]))
    
    tn, fp, fn, tp = confusion_matrix(dev['label'],pred).ravel()
    print('Historical')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current only after set 1 confusion matrix
    svc = model
    rfe = model_selection.getRFE(model,num_features)
    rfe = rfe.fit(after_set_1_train[curr],after_set_1_train['label'])
    svc.fit(rfe.transform(after_set_1_dev[curr]),after_set_1_dev['label'])
    pred = svc.predict(rfe.transform(after_set_1_dev[curr]))
    
    tn, fp, fn, tp = confusion_matrix(after_set_1_dev['label'],pred).ravel()
    print('Current only after set 1')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current + historical only after set 1 confusion matrix
    svc = model
    X = after_set_1_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    y = after_set_1_train['label']
    rfe = model_selection.getRFE(model,num_features)
    rfe = rfe.fit(X,y)
    svc.fit(rfe.transform(X),y)
    pred = svc.predict(rfe.transform(after_set_1_dev.drop(drop_col,axis = 1).dropna(axis=0, how='any')))
    
    tn, fp, fn, tp = confusion_matrix(after_set_1_dev['label'],pred).ravel()
    print('Current + historical after set 1')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))

    #current + historical only after set 2 confusion matrix
    svc = model
    X = after_set_2_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    y = after_set_2_train['label']
    rfe = model_selection.getRFE(model,15)
    rfe = rfe.fit(X,y)
    svc.fit(rfe.transform(X),y)
    pred = svc.predict(rfe.transform(after_set_2_dev.drop(drop_col,axis = 1).dropna(axis=0, how='any')))
    
    tn, fp, fn, tp = confusion_matrix(after_set_2_dev['label'],pred).ravel()
    print('Current + historical after set 2')
    print('tn, fp, fn, tp',(tn, fp, fn, tp))
    return

def outputRanking(model,feature, label,filename):
    rfe = model_selection.getRFE(model,1)
    ranking = model_selection.getRFERanking(rfe,feature, label)
    ranking.to_csv(filename)
    print('-----',filename,'-----')
    print(ranking)
    return ranking

def getData(diff):
    #read data with diff model
    label_, feature_ = joined_data_pred.importData(diff)
    data_ = joined_data_pred.joinLabelFeature(label_, feature_)
    drop_col = ['after_set']

    data = joined_data_pred.cleanDataForSklearn(data_)

    all_train, test = train_test_split(data, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    after_set_1_all_train = all_train[all_train['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_all_train = all_train[all_train['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    after_set_1_train = train[train['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_train = train[train['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    after_set_1_dev = dev[dev['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_dev = dev[dev['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    all_train = all_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    test = test.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    train = train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    dev = dev.drop(drop_col,axis = 1).dropna(axis=0, how='any')

    data_split = {}
    data_split['after_set_1_train'] =  after_set_1_train
    data_split['after_set_2_train'] =  after_set_2_train
    data_split['after_set_1_all_train'] =  after_set_1_all_train
    data_split['after_set_2_all_train'] =  after_set_2_all_train
    data_split['after_set_1_dev'] =  after_set_1_dev
    data_split['after_set_2_dev'] =  after_set_2_dev

    data_split['all_train'] =  all_train
    data_split['test'] =  test
 
    data_split['train'] =  train
    data_split['dev'] =  dev
    return data_split

def main():
    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct']
    
    data_split = getData(diff = True)
    '''
    #----GET CONFUSION MATRIX FOR BEST MODEL----#
    getConfusionMatrix(data_split,num_features = 15,model = LinearSVC())
    
    #----GET FEATURE RANKING FOR DIFFERENT DATA MODEL----#
    #historical + current 
    #after set 1 with logistic regression
    outputRanking(LogisticRegression(),data_split['after_set_1_all_train'].drop(['label'], axis=1),data_split['after_set_1_all_train']['label'],'log_after_set1_ranking_diff.csv')
    #after set 1 with svc
    outputRanking(LinearSVC(),data_split['after_set_1_all_train'].drop(['label'], axis=1),data_split['after_set_1_all_train']['label'],'svc_after_set1_ranking_diff.csv')
    #after set 2 with logistic
    outputRanking(LogisticRegression(),data_split['after_set_2_all_train'].drop(['label'], axis=1),data_split['after_set_2_all_train']['label'],'log_after_set2_ranking_diff.csv')
    #after set 2 with svc
    outputRanking(LinearSVC(),data_split['after_set_2_all_train'].drop(['label'], axis=1),data_split['after_set_2_all_train']['label'],'svc_after_set2_ranking_diff.csv')
    '''
    #historical only
    outputRanking(LogisticRegression(),data_split['train'][hist],data_split['train']['label'],'log_hist_ranking_diff.csv')
    #outputRanking(LinearSVC(),data_split['train'][hist],data_split['train']['label'],'log_hist_ranking_diff.csv')

    #current after set 1 only
    #outputRanking(LogisticRegression(),data_split['train'][curr],data_split['train']['label'],'log_hist_ranking_diff.csv')
    #outputRanking(LinearSVC(),data_split['train'][curr],data_split['train']['label'],'log_hist_ranking_diff.csv')


    return


if __name__ == '__main__':
    main()