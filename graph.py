import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import logistic
import model_selection
import svm

import joined_data_pred

def main():
    label_, feature_ = joined_data_pred.importData()

    data_ = joined_data_pred.joinLabelFeature(label_, feature_)

    # remove useless data cols from data_
    data = joined_data_pred.cleanDataForSklearn(data_)

    #split into train set, dev set, test set
    all_train, test = train_test_split(data, test_size=0.20, random_state=666)
    train, dev = train_test_split(all_train, test_size=0.25, random_state=666)

    after_set_1_all_train = all_train[all_train['after_set']==1]
    after_set_2_all_train = all_train[all_train['after_set']==2]


    return

if __name__ == '__main__':
	main()