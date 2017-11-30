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
    label_ = pd.read_csv('./data_scripts/result_per_match.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    feature_ = pd.read_csv('./data_scripts/final_joined_data.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    #print(label_.shape)
    #print(feature_.shape)
    return label_, feature_


def cleanDataForSklearn(d_):
    #print(list(d_))
    print(d_.shape)

    # TODO keep 1st_srv_pct
    drop_col = ['match_id', 'player', 'match_winner', 'Unnamed: 0_x', 'Unnamed: 0_y', 'Unnamed: 0.1', 'Tournament', '1st_srv_pct']
    d = d_.drop(drop_col,axis = 1).dropna(axis=0, how='any')
    
    print (d.shape)
    return d


def main():
    label_, feature_ = importData()

    data_ = joinLabelFeature(label_, feature_)

    # remove useless data cols from data_
    data = cleanDataForSklearn(data_)

    #split into train set, dev set, test set
    all_train, test = train_test_split(data, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    acc = logistic.logPredSet1Set2Diff(train,dev,['after_set', 'label'])
    print('log diff',acc)


    return

if __name__ == '__main__':
    main()