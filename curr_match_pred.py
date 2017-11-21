import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
import logistic


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
    return df

def main():
    #data processing
    match_label, curr_match_data,match_details = importData()
    match_data = joinDetailsMatch(curr_match_data,match_details)
    df = joinLabelFeature(match_label,match_data)
    drop_col = ['match_id','after_set','player','match_winner','Unnamed: 0_x','Unnamed: 0_y','label']

    #split into train set, dev set, test set
    all_train, test = train_test_split(df, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    #####LOGISTIC#####
    #on training only
    #get accuracy comparison for after 1 set and after 2 set for logistic regression
    acc1,acc2 = logistic.logPredSet1Set2Diff(train,drop_col)
    print('train',acc1,acc2)
    #get error analysis
    after_set_1 = train[train['after_set']==1]
    after_set_2 = train[train['after_set']==2]

    error_analysis_after_set_1 = logistic.getErrorAnalysis(after_set_1,drop_col)
    error_analysis_after_set_2 = logistic.getErrorAnalysis(after_set_2,drop_col)

    #print('after set 1',error_analysis_after_set_1)
    #print('after set 2',error_analysis_after_set_2)

    #on dev only
    #get accuracy comparison for after 1 set and after 2 set
    acc1,acc2 = logistic.logPredSet1Set2Diff(dev,drop_col)
    print('dev',acc1,acc2)

    #####SVM#####
    

    return

if __name__ == '__main__':
	main()