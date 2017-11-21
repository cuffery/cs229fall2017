import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression


def importData():
    match_label = pd.read_csv('./yubo_processing/result_per_match.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_data = pd.read_csv('./yubo_processing/current_match_data.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_details = pd.read_csv('./yi_processing/processed_match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    
    match_details['Tournament']=match_details['Tournament'].str.lower()
    #print('# of davis cup in matches',match_details[match_details['Tournament'].str.contains("davis cup")].shape[0])
    matches = match_details[match_details['Tournament'].str.contains("davis cup")==False]
    return match_label,match_data,match_details

def joinDetailsMatch(match_data,match_details):
    details = ['surface_hard','surface_grass','surface_clay','match_id','grand_slam']
    match = pd.merge(match_data,match_details[details],on="match_id",how = 'inner')
    return match

def joinLabelFeature(label,feature):
    df = pd.merge(label,feature,on="match_id",how = 'inner')
    df.loc[df['player'] == df['match_winner'], 'label'] = 1
    df.loc[df['player'] != df['match_winner'], 'label'] = 0




    #df[df['player'] == df['match_winner']]['label']=1
    #df[df['player'] != df['match_winner']]['label']=0
    #print(df['label'].iloc[1000])

    return df

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
    #data processing
    match_label, curr_match_data,match_details = importData()
    match_data = joinDetailsMatch(curr_match_data,match_details)
    df = joinLabelFeature(match_label,match_data)
    drop_col = ['match_id','after_set','player','match_winner','Unnamed: 0_x','Unnamed: 0_y','label']

    #get accuracy comparison for after 1 set and after 2 set
    acc1,acc2 = logPredSet1Set2Diff(df,drop_col)
    print(acc1,acc2)
    #get error analysis
    after_set_1 = df[df['after_set']==1]
    after_set_2 = df[df['after_set']==2]

    error_analysis_after_set_1 = getErrorAnalysis(after_set_1,drop_col)
    error_analysis_after_set_2 = getErrorAnalysis(after_set_2,drop_col)

    print('after set 1',error_analysis_after_set_1)
    print('after set 2',error_analysis_after_set_2)

    return

if __name__ == '__main__':
	main()