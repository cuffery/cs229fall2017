import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logistic
import svm

def joinLabelFeature(label,feature):
    df = pd.merge(label,feature, on = "match_id", how = 'inner')
    df.loc[df['player'] == df['match_winner'], 'label'] = 1
    df.loc[df['player'] != df['match_winner'], 'label'] = 0
    return df


def importData():
    # import data
    label_ = pd.read_csv('./yubo_processing/result_per_match.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    feature_ = pd.read_csv('./final_joined_data.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    #print(label_.shape)
    #print(feature_.shape)
    return label_, feature_

def cleanDataForSklearn(d_):
    print(list(d_))
    print(d_.shape)
    
    #drop_col = 

    return

def main():
    label_, feature_ = importData()

    data_ = joinLabelFeature(label_, feature_)

    # remove useless data cols data_
    data = cleanDataForSklearn(data_)

    # run models

    return

if __name__ == '__main__':
    main()