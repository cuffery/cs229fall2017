import pandas as pd
import numpy as np
import scipy
from sklearn import svm
import logistic
    
    
def getAccuracy(feature, label):
    clf = svm.SVC()
    clf.fit(feature,label)
    return clf.score(feature,label)

def getTestAccuracy(feature_train,label_train,feature_test,label_test):
    clf = svm.SVC()
    clf.fit(feature_train,label_train)
    return clf.score(feature_test,label_test)

def svmPredSet1Set2Diff(df_train,df_test,drop_col):
    feature1_train,label1_train,feature1_test,label1_test,feature2_train,label2_train,feature2_test,label2_test = logistic.assignTrainDev(df_train,df_test,drop_col)
    
    acc1_test = getTestAccuracy(feature1_train,label1_train,feature1_test,label1_test)
    acc2_test = getTestAccuracy(feature2_train,label2_train,feature2_test,label2_test)

    acc1_train = getAccuracy(feature1_train,label1_train)
    acc2_train = getAccuracy(feature2_train,label2_train)
    return {'train accuracy after set 1':acc1_train,'test accuracy after set 1':acc1_test,'train accuracy after set2': acc2_train,'test accuracy after set 2':acc2_test}

def getTrainingErrorAnalysis(df_train, drop_col):

    ls = []
    dropped = []
    label = df_train['label']
    feature = df_train.drop(drop_col,axis =1)
    #print(list(feature))
    #n = feature.shape[1]-1

    #get error analysis on training data after set 1
    for i in range(feature.shape[1],1,-1):
        #print(i)
        dropped.append(list(feature)[i-1])
        feature = feature.ix[:, 0:i]
        #feature = feature.drop(feature.columns[i], axis=1)
        ls.append(getAccuracy(feature,label))
    result = pd.DataFrame({'dropped_col_train': dropped,'accuracy_rate_train':ls})
    result['diff_training'] = result.accuracy_rate_train.diff()
    return result

def getTestingErrorAnalysis(df_train,df_test,drop_col):
    ls = []
    dropped = []
    label_train = df_train['label']
    feature_train = df_train.drop(drop_col,axis =1)
    label_test = df_test['label']
    feature_test = df_test.drop(drop_col,axis =1)
    #print(list(feature))
    #n = feature.shape[1]-1
    for i in range(feature_train.shape[1],1,-1):
        #print(i)
        dropped.append(list(feature_train)[i-1])
        feature_train = feature_train.ix[:, 0:i]
        feature_test = feature_test.ix[:, 0:i]

        #feature = feature.drop(feature.columns[i], axis=1)
        ls.append(getTestAccuracy(feature_train,label_train,feature_test,label_test))
    result = pd.DataFrame({'dropped_col_test': dropped,'accuracy_rate_test':ls})
    result['diff_test'] = result.accuracy_rate_test.diff()     
    return result

def main():
    
    return


if __name__ == '__main__':
    main()