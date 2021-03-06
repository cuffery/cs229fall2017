'''
  model_evaluation.py
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import operator
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import log_loss

#our scripts
import graph
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
    
    #get learning curve plots for linear svc  
    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff','avg_minutes_lost','avg_minutes_won','avg_age_diff_lost_to','avg_age_diff_won_against','best_of']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct','match_day_age','opponent_match_day_age','rally_sum']

    after_set_1_all_train = data_split['after_set_1_all_train']
    after_set_2_all_train = data_split['after_set_2_all_train']
    all_train = data_split['all_train']

    all_train = data_split['all_train']
    test = data_split['test']
    
    drop_col = ['label']
    '''
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
    '''

    drop_col = ['label']
    y = all_train['label']
    X = all_train[hist]

    rfe = model_selection.getRFE(LinearSVC(),num_features)
    rfe = rfe.fit(X,y)


    train_sizes, train_scores, valid_scores = learning_curve(SVC(), rfe.transform(X), y, train_sizes=np.linspace(.1, 1.0, 5), cv=5)

    print("train_sizes_lr\n", train_sizes)
    print("train_scores_lr\n", train_scores)
    print("valid_scores_lr\n", valid_scores)
    plot_learning_curve(train_sizes, train_scores, valid_scores, 'hist_linearsvc_learning_curve.png', 3)
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
    '''
    #remove newly added features
    drop_new_col = ['avg_minutes_lost','avg_minutes_won','avg_age_diff_lost_to','avg_age_diff_won_against','best_of','match_day_age','opponent_match_day_age','rally_sum']
    data = data.drop(drop_new_col,axis = 1).dropna(axis=0, how='any')
    '''
    all_train, test = train_test_split(data, test_size=0.20, random_state=521)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=521)

    after_set_1_all_train = all_train[all_train['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_all_train = all_train[all_train['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    after_set_1_train = train[train['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_train = train[train['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    after_set_1_dev = dev[dev['after_set']==1].drop(drop_col,axis = 1).dropna(axis=0, how='any')
    after_set_2_dev = dev[dev['after_set']==2].drop(drop_col,axis = 1).dropna(axis=0, how='any')

    all_train = all_train.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    #test = test.drop(drop_col,axis = 1).dropna(axis=0, how='any')
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

def getAccAlg(feature,label):
    #return best model
    accuracy = {}
    model = {}
    #Using RFE with logistic regression to get top 15 features 
    rfe = model_selection.getRFE(LogisticRegression(),15)
    rfe.fit(feature,label)
    feature_t = rfe.transform(feature)

    #Logstic
    '''
    model = logistic.getLogModel(feature_t,label)

    print('logistic',model.score(feature_dev_t,label_dev))
    '''
    model = logistic.getLogModel(feature_t,label)
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'logistic'
    accuracy[key] = score.mean()
    print('logistic',score.mean())
    
    #Using RFE with LinearSVC to get top 15 features 
    rfe = model_selection.getRFE(LinearSVC(),15)
    rfe.fit(feature,label)
    feature_t = rfe.transform(feature)
    
    #Linear SVC
    model = svm.getLinearSVCModel(feature_t,label)
    score = model_selection.getCrossValScore(model,feature_t,label,5)

    key = 'linear SVC'
    accuracy[key] = score.mean()
    print('linear SVC',score.mean())
    
    #SVC with different kernel with top 15 features
    model = svm.getSVCModel(feature_t,label,ker = 'rbf')
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'SVC + rbf kernel for 15 features'
    accuracy[key] = score.mean()
    print('SVC + rbf kernel for 15 features',score.mean())
    
    
    model = svm.getSVCModel(feature_t,label,ker = 'poly')
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'SVC + poly kernel for 15 features'
    accuracy[key] = score.mean()
    print('SVC + poly kernel for 15 features',score.mean())
    
    model = svm.getSVCModel(feature_t,label,ker = 'linear')
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'SVC + linear kernel for 15 features'
    accuracy[key] = score.mean()
    print('SVC + linear kernel for 15 features',score.mean())

    #SVC with different kernel with all features
    model = svm.getSVCModel(feature_t,label,ker = 'rbf')
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'SVC + rbf kernel for all features'
    accuracy[key] = score.mean()
    print('SVC + rbf kernel for all features',score.mean())
    
    
    model = svm.getSVCModel(feature_t,label,ker = 'poly')
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'SVC + poly kernel for all features'
    accuracy[key] = score.mean()
    print('SVC + poly kernel for all features',score.mean())
    
    model = svm.getSVCModel(feature_t,label,ker = 'linear')
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'SVC + linear kernel for all features'
    accuracy[key] = score.mean()
    print('SVC + linear kernel for all features',score.mean())
    

    #Naive Bayes
    #Using top 15 features
    model = GaussianNB()
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'SVC + linear kernel'
    accuracy[key] = score.mean()
    print('Naive Bayes with 15 features',score.mean())

    model = GaussianNB()
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'Naive Bayes with 15 features'
    accuracy[key] = score.mean()
    print('Naive Bayes with 15 features',score.mean())

    #Gaussian Discriminant Analysis
    model = QuadraticDiscriminantAnalysis()
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = "Quad discriminant anlysis with 15 features"
    accuracy[key] = score.mean()
    print('Quad discriminant anlysis with 15 features',score.mean())

    #Linear Gaussian Discriminant Analysis
    model = LinearDiscriminantAnalysis()
    score = model_selection.getCrossValScore(model,feature_t,label,5)
    key = 'Linear discriminant anlysis with 15 features'
    accuracy[key] =score.mean()
    print('Linear discriminant anlysis with 15 features',score.mean())

    #Using all features
    model = GaussianNB()
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'Naive Bayes with all features'
    accuracy[key] = score.mean()
    print('Naive Bayes with all features',score.mean())
    
    #Gaussian Discriminant Analysis
    model = QuadraticDiscriminantAnalysis()
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'Quad discriminant anlysis with all features'
    accuracy[key] = score.mean()
    print('Quad discriminant anlysis with all features',score.mean())
    
    #Linear Gaussian Discriminant Analysis
    model = LinearDiscriminantAnalysis()
    score = model_selection.getCrossValScore(model,feature,label,5)
    key = 'Linear discriminant anlysis with all features'
    accuracy[key] = score.mean()
    print('Linear discriminant anlysis with all features',score.mean())

    best_model = max(accuracy, key=accuracy.get)
    print("BEST MODEL IS: ",best_model)
    return best_model



def main():
    warnings.filterwarnings("ignore", category=Warning)

    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff','avg_minutes_lost','avg_minutes_won','avg_age_diff_lost_to','avg_age_diff_won_against','best_of']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct','match_day_age','opponent_match_day_age','rally_sum']
    
    data_split = getData(diff = False)
    print(list(data_split['train']))
    i = 24

    #get_learning_curve_plots(data_split,15)
    #raise Exception()
    result_model_hist = []
    result_model_curr = []
    result_model_hist_curr1 = []
    result_model_hist_curr2 = [] 
    max_iter = 25

    '''
    hist = ['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff']
    curr = ['surface_hard', 'surface_grass', 'surface_clay','aces', 'dfs', 'unforced', '1st_srv_pct', 'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct']
    '''
    '''
    #----FEATURE ANALYSIS AND LEARNING CURVE----#
    #Get graphs with feature analysis and learning curve
    #This script outputs graphs used in report and poster
    graph.main()
    '''
    while i < max_iter:
        #----EXPLORE DIFFERENT MODELS FOR FOUR DATA MODEL----#
        #Using hist + after 1st set
        print('-----Data model: using current + after 1st set-----')

        feature = data_split['after_set_1_all_train'].drop(['label'],axis = 1)
        label = data_split['after_set_1_all_train']['label']
        
        #getAccAlg(feature,label)

        print("We picked linearSVC with 15 features as best model")
        model = LinearSVC()
        rfe = model_selection.getRFE(LinearSVC(),15)
        rfe.fit(feature,label)
        feature_t = rfe.transform(feature)

        model.fit(feature_t,label)
        #get learning curve
        train_sizes, train_scores, valid_scores = learning_curve(model, feature_t, label, train_sizes=np.linspace(.1, 1.0, 5), cv=5)
        plot_learning_curve(train_sizes, train_scores, valid_scores, 'learning_curve_hist_curr1.png', 2)

        test = data_split['test']
        feature_test = test[test['after_set']==1].drop(['label','after_set'],axis = 1)
        label_test = test[test['after_set']==1]['label']

        print('Test score = ',model.score(rfe.transform(feature_test),label_test))
        result_model_hist_curr1.append(model.score(rfe.transform(feature_test),label_test))
        print(result_model_hist_curr1)

        #print confusion matrix for best model
        tn, fp, fn, tp = confusion_matrix(model.predict(rfe.transform(feature_test)),label_test).ravel()
        print('Hist + Current after set 1')
        print('tp, fp, tn, fp',(tp, fp, tn, fp))

        #Using hist + after 2nd set
        print('-----Data model: using current + after 2nd set-----')

        feature = data_split['after_set_2_all_train'].drop(['label'],axis = 1)
        label = data_split['after_set_2_all_train']['label']
        
        #getAccAlg(feature,label)

        print("We picked linearSVC with 15 features as best model")
        model = LinearSVC()
        rfe = model_selection.getRFE(LinearSVC(),15)
        rfe.fit(feature,label)
        feature_t = rfe.transform(feature)

        model.fit(feature_t,label)
        
        #get learning curve
        train_sizes, train_scores, valid_scores = learning_curve(model, feature_t, label, train_sizes=np.linspace(.1, 1.0, 5), cv=5)
        plot_learning_curve(train_sizes, train_scores, valid_scores, 'learning_curve_hist_curr2.png', 2)

        test = data_split['test']
        feature_test = test[test['after_set']==2].drop(['label','after_set'],axis = 1)
        label_test = test[test['after_set']==2]['label']

        print('Test score = ',model.score(rfe.transform(feature_test),label_test))
        result_model_hist_curr2.append(model.score(rfe.transform(feature_test),label_test))
        #print confusion matrix for best model
        tn, fp, fn, tp = confusion_matrix(model.predict(rfe.transform(feature_test)),label_test).ravel()
        print('Hist + Current after set 2')
        print('tp, fp, tn, fp',(tp, fp, tn, fp))
        
        #Using historical only
        print('-----Data model: historical data only-----')

        feature = data_split['all_train'][hist]
        label = data_split['all_train']['label']
        
        #getAccAlg(feature,label)

        print("We picked SVC with rbf kernel with all features as best model")
        print('with rfe selection')

        # We learnt that the behavior of SVC-rbf kernnel is very sensitive to the gamma and C parameter.
        # Intuitively, the gamma parameter defines how far the influence of a single training 
        # example reaches, with low values meaning ‘far’ and high values meaning ‘close’.
        # The C parameter trades off misclassification of training examples against simplicity
        #  of the decision surface.

        # These gamma = 0.25 and C = 0.27 values for rbf kernnel was found through experiments, 
        # with the guiding idea of "good models can be found on a diagonal of C and gamma" from
        # previous research
        model = SVC(kernel = 'rbf', gamma = 0.25, C = 0.27)

        #get learning curve
        train_sizes, train_scores, valid_scores = learning_curve(model, feature, label, train_sizes=np.linspace(.1, 1.0, 5), cv=5)

        plot_learning_curve(train_sizes, train_scores, valid_scores, 'learning_curve_hist.png', 2)

        model.fit(feature,label)

        print('FEATURE FLAG', list(feature))

        test = data_split['test']
        feature_test = test[hist]
        label_test = test['label']
        
        print('Test score = ',model.score(feature_test,label_test))
        result_model_hist.append(model.score(feature_test,label_test))

        tn, fp, fn, tp = confusion_matrix(model.predict(feature_test),label_test).ravel()
        print('Historical only')
        print('tp, fp, tn, fp',(tp, fp, tn, fp))
        
        '''
        print('with all features')
        rfe = model_selection.getRFE(LinearSVC(),15)
        rfe.fit(feature,label)
        feature_t = rfe.transform(feature)
        model.fit(feature_t,label)
        print('Test score = ',model.score(rfe.transform(feature_test),label_test))
        result_model_hist.append(model.score(rfe.transform(feature_test),label_test))
        '''
        
        #Using current after set 1 only
        print('-----Data model: current data only-----')

        feature = data_split['after_set_1_all_train'][curr]
        label = data_split['after_set_1_all_train']['label']    
        
        #getAccAlg(feature,label)
        
        print("We picked logistic regression with 15 features as best model")
        model = LogisticRegression()

        rfe = model_selection.getRFE(LogisticRegression(),15)
        rfe.fit(feature,label)
        feature_t = rfe.transform(feature)

        model.fit(feature_t,label)
        train_sizes, train_scores, valid_scores = learning_curve(model, feature_t, label, train_sizes=np.linspace(.1, 1.0, 5), cv=5)
        plot_learning_curve(train_sizes, train_scores, valid_scores, 'learning_curve_curr1.png', 2)

        test = data_split['test']
        feature_test = test[test['after_set']==1][curr]
        label_test = test[test['after_set']==1]['label']
        print('Test score = ',model.score(rfe.transform(feature_test),label_test))
        result_model_curr.append(model.score(rfe.transform(feature_test),label_test))

        #print confusion matrix for best model
        tn, fp, fn, tp = confusion_matrix(model.predict(rfe.transform(feature_test)),label_test).ravel()
        print('Current after set 1')
        print('tp, fp, tn, fp',(tp, fp, tn, fp))

        i += 1
    print("test accuracy for historical only")
    print(result_model_hist)
    print("test accuracy for current only")
    print(result_model_curr)
    print("test accuracy for current set 1 + historical")
    print(result_model_hist_curr1)
    print("test accuracy for current set 2 + historical")
    print(result_model_hist_curr2)
    #----GET CONFUSION MATRIX FOR BEST MODEL----#
    #getConfusionMatrix(data_split,num_features = 15,model = LinearSVC())
    
    '''
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
    
    

    #historical only
    outputRanking(LogisticRegression(),data_split['train'][hist],data_split['train']['label'],'log_hist_ranking_diff.csv')
    outputRanking(LinearSVC(),data_split['train'][hist],data_split['train']['label'],'log_hist_ranking_diff.csv')

    #current after set 1 only
    outputRanking(LogisticRegression(),data_split['train'][curr],data_split['train']['label'],'log_hist_ranking_diff.csv')
    outputRanking(LinearSVC(),data_split['train'][curr],data_split['train']['label'],'log_hist_ranking_diff.csv')
    
    '''
    return


if __name__ == '__main__':
    main()