from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

import pandas as pd
import numpy as np
import scipy

def getAccuracy(feature,label):
    #old function that computes the regression
    logreg = LogisticRegression()
    logreg.fit(feature,label)
    return logreg.score(feature,label)

def getAcc(model,feature,label):
    #get accuracy from predetermined model, on new or same features/labels
    return model.score(feature,label)

def getLogModel(feature, label):
    #returns fitted model of logistic regression
    logreg = LogisticRegression()
    model = logreg.fit(feature,label)
    return model

def getLogLoss(model,feature,label):
    #get log loss from predetermined model, on new or same features/labels
    prediction = model.predict(feature)
    return log_loss(label, prediction)

def getTestAccuracy(feature_train,label_train,feature_test,label_test):
    #train, dev = train_test_split(feature, test_size=0.25, random_state=666)
    logreg = LogisticRegression()
    logreg.fit(feature_train,label_train)
    return logreg.score(feature_test,label_test)

def assignTrainDev(df_train, df_test,drop_col):
    after_set_1_train = df_train[df_train['after_set']==1]
    after_set_2_train = df_train[df_train['after_set']==2]

    label1_train=after_set_1_train['label']
    label2_train=after_set_2_train['label']

    feature1_train = after_set_1_train.drop(drop_col,axis = 1)
    feature2_train = after_set_2_train.drop(drop_col,axis = 1)

    after_set_1_test = df_test[df_test['after_set']==1]
    after_set_2_test = df_test[df_test['after_set']==2]

    label1_test=after_set_1_test['label']
    label2_test=after_set_2_test['label']

    feature1_test = after_set_1_test.drop(drop_col,axis = 1)
    feature2_test = after_set_2_test.drop(drop_col,axis = 1)
    return feature1_train,label1_train,feature1_test,label1_test,feature2_train,label2_train,feature2_test,label2_test

def logPredSet1Set2Diff(df_train, df_test, drop_col):
    #returns test accuracy
    feature1_train,label1_train,feature1_test,label1_test,feature2_train,label2_train,feature2_test,label2_test = assignTrainDev(df_train,df_test,drop_col)
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