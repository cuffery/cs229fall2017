#   hist_match_pred.py
#   
#   Script that runs some very preliminary ML models based on historical ATP Matches data only
#
#   Created by: Eddie Chen
#	Last edit: 11/21/2017 17:07

import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import logistic
import svm


def importData():

	data = pd.read_csv("./ec_processing/processed.csv",
					 index_col=None,
					 header=0,
					 encoding = "ISO-8859-1")

	data = data.drop('match_num',axis=1) 
	data = data.drop('match_result',axis=1)
	data = data.drop('opponent_ID',axis=1)
	data = data.drop('player_ID',axis=1) ##if I generate afresh, I dont need this line
	data = data.drop('Date',axis=1)
	data = data.drop('player_2_id',axis=1)
	data = data.drop('player_1_id',axis=1)
	print(data.shape)

	return data

def LogRegGetAccuracy(X,y):
	#split into train, test
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=233)
	logreg = LogisticRegression()
	logreg.fit(X_train,y_train)
	return logreg.score(X_test,y_test)


def LogRegGetErrorAnalysis(X,y):
	ls = []
	dropped = []
	label = y
	feature = X
	#print(list(feature))
	#n = feature.shape[1]-1
	
	#print(list(feature))
	for i in range(feature.shape[1],1,-1):
		#print(i)
		dropped.append(list(feature)[i-1])
		feature = feature.ix[:, 0:i]
		#feature = feature.drop(feature.columns[i], axis=1)
		ls.append(LogRegGetAccuracy(feature,label))
	result = pd.DataFrame({'dropped_col': dropped,'accuracy_rate':ls})
	result['diff'] = result.accuracy_rate.diff()
	return result

def main():
	#data processing
	data = importData()
	trimmed = data.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
	print(trimmed.shape)
	# # print(trimmed.head())
	trimmed = trimmed.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	print(trimmed.shape) ##(1004, 24)

	## now load label and features
	y = trimmed['label']
	print(y.shape)

	X = trimmed.copy()
	X = X.drop('label', axis = 1)
	print(X.shape)

	
	error_analysis = LogRegGetErrorAnalysis(X,y)
	print('Logistic Regression Error Analysis',error_analysis)
	
	return 

if __name__ == '__main__':
	main()