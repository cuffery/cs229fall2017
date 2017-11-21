from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import scipy

def getAccuracy(feature,label):
    logreg = LogisticRegression()
    logreg.fit(feature,label)
    return logreg.score(feature,label)

def logPredSet1Set2Diff(df,drop_col):
    logreg1 = LogisticRegression()
    logreg2 = LogisticRegression()

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
    return result

def main():

    return

if __name__ == '__main__':
	main()