#   PreliminaryTest.py
#   
#   Script that runs some very preliminary ML models based on ATP Matches data only
#
#   Created by: Eddie Chen
#	Last edit: 11/20/2017 15:07


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import datetime
import CommonOpponentStats as COS
from sklearn import linear_model

def prepData():
	atpmatches = COS.readATPMatchesParseTime("../tennis_atp")
	atpmatches.to_csv('atp_2000s_matches_master.csv', sep=",",encoding = "ISO-8859-1")

	return

def generateTrainingData():
	test = pd.read_csv('atp_2000s_matches_master.csv', index_col=None,
		header=0,encoding = "ISO-8859-1")
	# print(test.shape) ## (50219, 50)
	results = pd.DataFrame()
	res = pd.DataFrame()

	container = list()
	for i in xrange(test.shape[0]):
		p1 = test.winner_id[i]
		p2 = test.loser_id[i]
		d = test.tourney_date[i]
		y = 1
		p1opponents = COS.FindOpponents(p1, test, d)
		p2opponents = COS.FindOpponents(p2, test, d)
		(op1, op2) = COS.FindCommonOpponentStats(p1opponents, p2opponents)
		if (op1.shape[0] ==0 or op2.shape[0] == 0):
			## check if we have any common opponent record before the match date, use date constraint
			## if not, use player's historical performance regardless of common opponent
			avgp1 = COS.ComputeHistoricalAvg(p1, d, p1opponents)
			avgp2 = COS.ComputeHistoricalAvg(p2, d, p2opponents)
			if (avgp1.shape[0] == 0 or avgp2.shape[0] ==0):
				# res = pd.DataFrame(np.nan,index =xrange(1),columns = xrange(25))
				res = pd.DataFrame()
				res['no_historical_data'] = 1
			# 	res = ##preserve column headers, fill with NA
			if (avgp1.shape[0] > 0 and avgp2.shape[0] > 0):
				res = COS.ComputeAvgFromCommonOpp(avgp1, avgp2)
				res['no_common_opponent'] = 1
		else:
			## else, compute using common opponent
			avgp1 = COS.ComputeHistoricalAvg(p1, d, op1)
			avgp2 = COS.ComputeHistoricalAvg(p2, d, op2)
			res = COS.ComputeAvgFromCommonOpp(avgp1, avgp2)
			res['no_common_opponent'] = 0
			res['no_historical_data'] = 0

		res['player_1_id'] = p1
		res['player_2_id'] = p2
		res['Date'] = d
		res['label'] = 1
		#container.append((p1,p2,d,res))
		results = results.append(res)
	return results
	# return

# main()
def main():

	# prepData()
	res = generateTrainingData()
	res.to_csv('processed.csv', sep=",",encoding = "ISO-8859-1")
	## data setup: winner labeled 1 - winner's stats 
	## loser labeled as 0 - negate all the stats
	## get raw data from comoon_test.csv

	## Cleaning data, ridding NaN
	

	# data = pd.read_csv('common_test.csv',
 #                        index_col=None,
 #                        header=0,encoding = "ISO-8859-1")

	# print(data.shape)

	# trimmed = data.dropna(axis=1, how='all', thresh=None, subset=None, inplace=False)
	# # print(trimmed.shape)
	# # print(trimmed.head())
	# trimmed = trimmed.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
	# print(trimmed.shape) ##(1004, 24)

	## now transform label - 1 if Player 1 wins, 0 otherwise

	return

if __name__ == '__main__':
	main()