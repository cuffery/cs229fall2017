import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logistic
import svm


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
    
    after_set_1_train = train[train['after_set']==1]
    after_set_2_train = train[train['after_set']==2]
    after_set_1_dev = dev[dev['after_set']==1]
    after_set_2_dev = dev[dev['after_set']==2]
    '''
    #####LOGISTIC#####
    #on training only
    #get training and dev set accuracy comparison for after 1 set and after 2 set for logistic regression
    acc = logistic.logPredSet1Set2Diff(train,dev,drop_col)
    print('log diff',acc)
    
    #get error analysis
    training_error_analysis_after_set_1 = logistic.getTrainingErrorAnalysis(after_set_1_train,drop_col)
    training_error_analysis_after_set_2 = logistic.getTrainingErrorAnalysis(after_set_2_train,drop_col)
    testing_error_analysis_after_set_1 = logistic.getTestingErrorAnalysis(after_set_1_train,after_set_1_dev,drop_col)
    testing_error_analysis_after_set_2 = logistic.getTestingErrorAnalysis(after_set_2_train,after_set_2_dev,drop_col)

    print('training error analysis after set 1',training_error_analysis_after_set_1)
    print('training error analysis after set 2',training_error_analysis_after_set_2)
    print('testing error analysis after set 1',testing_error_analysis_after_set_1)
    print('testing error analysis after set 2',testing_error_analysis_after_set_2)
    '''
    #####SVM#####
    acc = svm.svmPredSet1Set2Diff(train,dev,drop_col)
    print('svm diff',acc)
    training_error_analysis_after_set_1 = svm.getTrainingErrorAnalysis(after_set_1_train,drop_col)
    training_error_analysis_after_set_2 = svm.getTrainingErrorAnalysis(after_set_2_train,drop_col)
    testing_error_analysis_after_set_1 = svm.getTestingErrorAnalysis(after_set_1_train,after_set_1_dev,drop_col)
    testing_error_analysis_after_set_2 = svm.getTestingErrorAnalysis(after_set_2_train,after_set_2_dev,drop_col)
    
    print('training error analysis after set 1',training_error_analysis_after_set_1)
    print('training error analysis after set 2',training_error_analysis_after_set_2)
    print('testing error analysis after set 1',testing_error_analysis_after_set_1)
    print('testing error analysis after set 2',testing_error_analysis_after_set_2)
    return

if __name__ == '__main__':
	main()