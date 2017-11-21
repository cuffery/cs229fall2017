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

def svmPredSet1Set2Diff(df,drop_col):
    after_set_1 = df[df['after_set']==1]
    after_set_2 = df[df['after_set']==2]

    label1=after_set_1['label']
    label2=after_set_2['label']

    feature1 = after_set_1.drop(drop_col,axis = 1)
    feature2 = after_set_2.drop(drop_col,axis = 1)

    acc1 = getAccuracy(feature1,label1)
    acc2 = getAccuracy(feature2,label2)
    return acc1,acc2

def getErrorAnalysis(df,drop_col):
    ls = []
    dropped = []
    label = df['label']
    feature = df.drop(drop_col,axis =1)
    #print(list(feature))
    #n = feature.shape[1]-1
    
    #print(list(feature))
    for i in range(feature.shape[1],1,-1):
        #print(i)
        dropped.append(list(feature)[i-1])
        feature = feature.ix[:, 0:i]
        #feature = feature.drop(feature.columns[i], axis=1)
        ls.append(getAccuracy(feature,label))
    result = pd.DataFrame({'dropped_col': dropped,'accuracy_rate':ls})
    result['diff'] = result.accuracy_rate.diff()
    print(result)
    return result

def main():
    
    return


if __name__ == '__main__':
    main()