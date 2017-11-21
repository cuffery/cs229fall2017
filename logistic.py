from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import scipy

def getAccuracy(feature,label):
    logreg = LogisticRegression()
    logreg.fit(feature,label)
    return logreg.score(feature,label)

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
    return {'test accuracy after set 1':acc1_test,'test accuracy after set 2':acc2_test,'train accuracy after set 1':acc1_train,'train accuracy after set2': acc2_train}

def getTrainingErrorAnalysis(df,drop_col):
    ls = []
    dropped = []
    label = df['label']
    feature = df.drop(drop_col,axis =1)
    #print(list(feature))
    #n = feature.shape[1]-1
    for i in range(feature.shape[1],1,-1):
        #print(i)
        dropped.append(list(feature)[i-1])
        feature = feature.ix[:, 0:i]
        #feature = feature.drop(feature.columns[i], axis=1)
        ls.append(getAccuracy(feature,label))
    result = pd.DataFrame({'dropped_col': dropped,'accuracy_rate':ls})
    result['diff'] = result.accuracy_rate.diff()     
    return result

def getTestingErrorAnalysis(df_train,df_test,drop_col):
    ls = []
    dropped = []
    label = df['label']
    feature = df.drop(drop_col,axis =1)
    #print(list(feature))
    #n = feature.shape[1]-1
    for i in range(feature.shape[1],1,-1):
        #print(i)
        dropped.append(list(feature)[i-1])
        feature = feature.ix[:, 0:i]
        #feature = feature.drop(feature.columns[i], axis=1)
        ls.append(getAccuracy(feature,label))
    result = pd.DataFrame({'dropped_col': dropped,'accuracy_rate':ls})
    result['diff'] = result.accuracy_rate.diff()     
    return result

def main():

    return

if __name__ == '__main__':
	main()