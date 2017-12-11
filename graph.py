import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import model_evaluation
import logistic
import model_selection
import svm

import joined_data_pred

def main():
    #-----PREPARE DATA FOR PLAYER DATA-----#
    data_split = model_evaluation.getData(diff = False)

    after_set_1_train= data_split['after_set_1_train']
    after_set_2_train = data_split['after_set_2_train']
    after_set_1_all_train = data_split['after_set_1_all_train']
    after_set_2_all_train = data_split['after_set_2_all_train']
    after_set_1_dev = data_split['after_set_1_dev']
    after_set_2_dev = data_split['after_set_2_dev']

    all_train = data_split['all_train']
    test = data_split['test']

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
    fig.suptitle('RFE using player model for current match data')

    plt.savefig('Error Comparison with Feature Size for Logistic and SVC for player data.png')
    
    
    #-----PREPARE DATA FOR DIFF DATA-----#
    data_split = model_evaluation.getData(diff = True)

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
    
    fig.suptitle('RFE using difference model for current match data')

    plt.savefig('Error Comparison with Feature Size for Logistic and SVC for diff data.png')

    
    #-----GRAPH PRE-MATCH V. IN-MATCH PREDICTION-----#
    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct', 'ttl_pts_won']
    
    df_log_hist = model_selection.getRFEMeanSTD(LogisticRegression(),all_train[hist],all_train['label'],'Logistic Regression Hist')
    df_svc_hist = model_selection.getRFEMeanSTD(LinearSVC(),all_train[hist],all_train['label'],'Linear SVC Hist')
    df_log_curr_1 = model_selection.getRFEMeanSTD(LogisticRegression(),after_set_1_all_train[curr],after_set_1_all_train['label'],'Logistic Regression Curr')
    df_svc_curr_1 = model_selection.getRFEMeanSTD(LinearSVC(),after_set_1_all_train[curr],after_set_1_all_train['label'],'Linear SVC Curr')

    fig, axs = plt.subplots(ncols=2, sharex=False)
    ax = axs[0]
    ax.errorbar(df_log_hist['Feature #'], df_log_hist['Mean Score'], yerr=df_log_hist['St.Dev'],label = "Logistic Regression",fmt = 'o')
    ax.errorbar(df_svc_hist['Feature #'], df_svc_hist['Mean Score'], yerr=df_svc_hist['St.Dev'],label = "Linear SVC",fmt = 'o')
    ax.set_ylim([0.5,1])
    ax.set_xlim([0,20])
    ax.xlabel('# of features selected with RFE')
    ax.ylabel('Accuracy')

    ax.legend(loc = 1)
    ax.set_title('Prematch: Match level historical')

    ax = axs[1]
    ax.errorbar(df_log_curr_1['Feature #'], df_log_curr_1['Mean Score'], yerr=df_log_curr_1['St.Dev'],label = "Logistic Regression",fmt = 'o')
    ax.errorbar(df_svc_curr_1['Feature #'], df_svc_curr_1['Mean Score'], yerr=df_svc_curr_1['St.Dev'],label = "Linear SVC",fmt = 'o')
    ax.set_ylim([0.5,1])
    ax.set_xlim([0,10])
    ax.xlabel('# of features selected with RFE')
    ax.ylabel('Accuracy')

    ax.legend(loc = 1)
    ax.set_title('In-Match: After 1st set')

    fig.savefig('Prediction hist curr.png')
    
    #-----GRAPH LEARNING CURVE----#
    model_evaluation.get_learning_curve_plots(diff = True)

    return

if __name__ == '__main__':
	main()