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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# util function
def parse(t):
	ret = []
	for ts in t:
		try:
			string = str(ts)
			tsdt = datetime.date(int(string[:4]), int(string[4:6]), int(string[6:]))
		except TypeError:
			tsdt = datetime.date(1900,1,1)
		ret.append(tsdt)
	return ret


def prepData():
	atpmatches = COS.readATPMatchesParseTime("../tennis_atp")
	atpmatches.to_csv('atp_2000s_matches_master.csv', sep=",",encoding = "ISO-8859-1")

	return

def generateTrainingData():
	atpmatches = COS.readATPMatchesParseTime("../tennis_atp")
	# atpmatches = pd.read_csv("atp_2000s_matches_master.csv",index_col=None,header=0)
	test = pd.read_csv("../tennis_atp/atp_matches_2010.csv",
						 index_col=None,
						 header=0,
						 parse_dates=[5],
						 encoding = "ISO-8859-1",
						 date_parser=lambda t:parse(t)) 
	# test = test.head(100)
	# print(test.shape) ## (50219, 50) (3058, 49)
	results = pd.DataFrame()
	res = pd.DataFrame()
	# p1 = test.winner_id[1]
	# p2 = test.loser_id[1]
	# d = test.tourney_date[1]
	# print(p1)
	# print(p2)
	# print(d)
	# return
	
	container = list()
	for i in xrange(test.shape[0]):
		if i % 100 == 0:
			print i
		p1 = test.winner_id[i]
		p2 = test.loser_id[i]
		d = test.tourney_date[i]
		y = 1
		p1opponents = COS.FindOpponents(p1, atpmatches, d)
		p2opponents = COS.FindOpponents(p2, atpmatches, d)
		(op1, op2) = COS.FindCommonOpponentStats(p1opponents, p2opponents)
		if (op1.shape[0] ==0 or op2.shape[0] == 0):
			## check if we have any common opponent record before the match date, use date constraint
			## if not, use player's historical performance regardless of common opponent
			avgp1 = COS.ComputeHistoricalAvg(p1, d, p1opponents)
			avgp2 = COS.ComputeHistoricalAvg(p2, d, p2opponents)
			# if (avgp1.shape[0] == 0 or avgp2.shape[0] ==0):
			# 	# res = pd.DataFrame(np.nan,index =xrange(1),columns = xrange(25))
			# 	res = pd.DataFrame()
			# 	res['no_historical_data'] = 1
			# 	res = ##preserve column headers, fill with NA
			if (avgp1.shape[0] > 0 and avgp2.shape[0] > 0):
				res = COS.ComputeAvgFromCommonOpp(avgp1, avgp2)
				# res['no_common_opponent'] = 1
		else:
			## else, compute using common opponent
			avgp1 = COS.ComputeHistoricalAvg(p1, d, op1)
			avgp2 = COS.ComputeHistoricalAvg(p2, d, op2)
			res = COS.ComputeAvgFromCommonOpp(avgp1, avgp2)
			# res['no_common_opponent'] = 0
			# res['no_historical_data'] = 0

		res['player_1_id'] = p1
		res['player_2_id'] = p2
		res['label'] = 1

		res_copy = pd.DataFrame({'match_result':1-res.match_result,
							'match_num':res.match_num, 
							'duration_minutes': res.duration_minutes,
							'label':0,
							# 'player_ID':res.opponent_ID,
							# 'opponent_ID':res.player_ID,
							'same_handedness': 1-res.same_handedness,
							'rank_diff':-res.rank_diff, # we use winner-loser, so likely a negative number
							'rank_pts_diff':-res.rank_pts_diff,
							'height_diff': -res.height_diff, 
							'first_serve_perc_diff': -res.first_serve_perc_diff, #(first_in/serve_pts)First serve percentage
							'ace_diff': -res.ace_diff, 
							'df_diff': -res.df_diff, # double faults
							'1st_serve_pts_won_diff': -res['1st_serve_pts_won_diff'], #(first_won/first_in) Percentage of points won on first serve
							'2nd_serve_pts_won_diff': -res['2nd_serve_pts_won_diff'],
							'bp_faced_diff':-res.bp_faced_diff, # break points faced
							'bp_saved_diff':-res.bp_saved_diff, # break points saved
							'bp_saving_perc_diff': -res.bp_saving_perc_diff,
							'SvGms_diff':-res.SvGms_diff, # serve game diff
							'svpt_diff': -res.svpt_diff # this tracks return points
							})

		res['Date'] = d
		res_copy['Date'] = d
		res_copy['player_1_id'] = p2
		res_copy['player_2_id'] = p1
		
		res = res.append(res_copy)
		results = results.append(res)
	# print(res.dtypes)
	return results
	# return
	

# main()
def main():

	## prepData()
	# res = generateTrainingData()

	## to save time, we just load from csv
	# res.to_csv('processed.csv', sep=",",encoding = "ISO-8859-1")
	data = pd.read_csv("processed.csv",
						 index_col=None,
						 header=0,
						 encoding = "ISO-8859-1")
	data = data.drop('match_num',axis=1) 
	data = data.drop('match_result',axis=1)
	data = data.drop('opponent_ID',axis=1)
	data = data.drop('player_ID',axis=1)
	data = data.drop('Date',axis=1)
	print(data.shape)
	## Cleaning data, ridding NaN
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


	## do test - training split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=233)

	print(y_train.shape)
	print(X_test.shape)

	## Now, create a Logistic Regression model
	regr = linear_model.LogisticRegression()
	regr.fit(X_train, y_train)
	y_pred = regr.predict(X_test)

	# The scores
	print('score: %.2f' % regr.score(X_test, y_test))

	# Plot outputs
	# plt.scatter(X_test, y_test,  color='black')
	# plt.plot(X_test, y_pred, color='blue', linewidth=3)

	# plt.xticks(())
	# plt.yticks(())

	# plt.show()

	return

if __name__ == '__main__':
	main()