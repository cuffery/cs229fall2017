#   CommonOpponentStats.py
#   
#   Script that takes in two players and outputs their stats calcualted through common opponent
#
#   Created by: Eddie Chen
#	Last edit: 11/18/2017 21:07

# Fatigue
# Weather
# Current match
# Injury


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import glob
import datetime

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

def parse_date(td):
	"""helper function to parse time"""
	resYear = float(td.days)/364.0                   # get the number of years including the the numbers after the dot
	resMonth = int((resYear - int(resYear))*364/30)  # get the number of months, by multiply the number after the dot by 364 and divide by 30.
	resYear = int(resYear)
	return str(resYear) + "y" + str(resMonth) + "m"

def readATPMatches(dirname):
	"""Reads ATP matches but does not parse time into datetime object"""
	allFiles = glob.glob(dirname + "/atp_matches_" + "20??.csv") ##restrict training set to matches from 2000s
	matches = pd.DataFrame()
	container = list()
	for filen in allFiles:
		df = pd.read_csv(filen,
						 index_col=None,
						 header=0)
		container.append(df)
	matches = pd.concat(container)
	return matches

def readATPMatchesParseTime(dirname):
	"""Reads ATP matches and parses time into datetime object"""
	allFiles = glob.glob(dirname + "/atp_matches_" + "20??.csv")
	allFiles = allFiles[:-1] ## avoid 2017 since its incomplete
	matches = pd.DataFrame()
	container = list()
	for filen in allFiles:
		df = pd.read_csv(filen,
						 index_col=None,
						 header=0,
						 parse_dates=[5],
						 encoding = "ISO-8859-1",
						 date_parser=lambda t:parse(t)) ##errored out here
		container.append(df)
	matches = pd.concat(container)
	return matches

## This returns a list of common opponents between p1 and p2
# def FindCommonOpponents(p1opp, p2opp):

# 	return

## This calculates differential stats from common opponents
def FindCommonOpponentStats(p1opp, p2opp):
	oppForP1 = pd.DataFrame(p1opp[p1opp.opponent_ID.isin(p2opp.opponent_ID)])
	oppForP2 = pd.DataFrame(p2opp[p2opp.opponent_ID.isin(p1opp.opponent_ID)])
	# print(oppForP1.opponent_name)
	# oppCount = 0
	return(oppForP1, oppForP2)

def getWinnerStats(df):
	df_trim = pd.DataFrame({'tourney_id':df.tourney_id, 
							'tourney_name':df.tourney_name, 
							'surface':df.surface, 
							'tourney_date':df.tourney_date, 
							'match_num':df.match_num, 
							'duration_minutes': df.minutes,
							'match_result':1.,
							'player_name':df.winner_name,
							'player_ID':df.winner_id,
							'opponent_name':df.loser_name,
							'opponent_ID':df.loser_id,
							'same_handedness': df.winner_hand == df.loser_hand, # track handedness
							'rank_diff':df.winner_rank - df.loser_rank, # we use winner-loser, so likely a negative number
							'rank_pts_diff':df.winner_rank_points - df.loser_rank_points,
							'height_diff': df.winner_ht - df.loser_ht, 
							'first_serve_perc_diff': df.w_1stIn*1.0/df.w_svpt - df.l_1stIn*1.0/df.l_svpt, #(first_in/serve_pts)First serve percentage
							'ace_diff': df.w_ace - df.l_ace, 
							'df_diff': df.w_df - df.l_df, # double faults
							'1st_serve_pts_won_diff': df.w_1stWon*1.0/df.w_1stIn - df.l_1stWon*1.0/df.l_1stIn, #(first_won/first_in) Percentage of points won on first serve
							'2nd_serve_pts_won_diff': df.w_2ndWon*1.0/(df.w_svpt - df.w_1stIn) - df.l_2ndWon*1.0/(df.l_svpt - df.l_1stIn),
							'bp_faced_diff':df.w_bpFaced - df.l_bpFaced, # break points faced
							'bp_saved_diff':df.w_bpSaved - df.l_bpSaved, # break points saved
							'bp_saving_perc_diff': df.w_bpSaved*1.0/df.w_bpFaced - df.l_bpSaved*1.0/df.l_bpFaced,
							'SvGms_diff':df.w_SvGms - df.l_SvGms, # serve game diff
							'svpt_diff': df.w_svpt - df.l_svpt # this tracks return points
							})
	return(df_trim)

def getLoserStats(df):
	df_trim = pd.DataFrame({'tourney_id':df.tourney_id, 
							'tourney_name':df.tourney_name, 
							'surface':df.surface, 
							'tourney_date':df.tourney_date, 
							'match_num':df.match_num, 
							'duration_minutes': df.minutes,
							'match_result':0.,
							'player_name':df.loser_name,
							'player_ID':df.loser_id,
							'opponent_name':df.winner_name,
							'opponent_ID':df.winner_id,
							'same_handedness': df.loser_hand == df.winner_hand, # track handedness
							'rank_diff':df.loser_rank - df.winner_rank, # we use winner-loser, so likely a negative number
							'rank_pts_diff':df.loser_rank_points - df.winner_rank_points,
							'height_diff': df.loser_ht - df.winner_ht, 
							'first_serve_perc_diff': df.l_1stIn*1.0/df.l_svpt - df.w_1stIn*1.0/df.w_svpt, #(first_in/serve_pts)First serve percentage
							'ace_diff': df.l_ace - df.w_ace, 
							'df_diff': df.l_df - df.w_df, # double faults
							'1st_serve_pts_won_diff': df.l_1stWon*1.0/df.l_1stIn - df.w_1stWon*1.0/df.w_1stIn, #(first_won/first_in) Percentage of points won on first serve
							'2nd_serve_pts_won_diff': df.l_2ndWon*1.0/(df.l_svpt - df.l_1stIn) - df.w_2ndWon*1.0/(df.w_svpt - df.w_1stIn),
							'bp_faced_diff':df.l_bpFaced - df.w_bpFaced, # break points faced
							'bp_saved_diff':df.l_bpSaved - df.w_bpSaved, # break points saved
							'bp_saving_perc_diff': df.l_bpSaved*1.0/df.l_bpFaced - df.w_bpSaved*1.0/df.w_bpFaced,
							'SvGms_diff':df.l_SvGms - df.w_SvGms, # serve game diff
							'svpt_diff': df.l_svpt - df.w_svpt # this tracks return points
							})
	return df_trim


## This returns a list of opponents for player p
def FindOpponents(p, matches, matchdate):
## find games where p won, and then where p lost (by player ID)
## we need to alter output format, so that it will in the format of:
## p, opp, tourney_id, surface, turney-level, p's stats - opp's stats for each feature
	names = pd.DataFrame()
	df = matches[matches.winner_id == p]
	df = df[df.tourney_date < matchdate]
	df_trim = getWinnerStats(df)
	opponents = list()
	opponents.append(df_trim)
	df = matches[matches.loser_id == p]
	df = df[df.tourney_date < matchdate]
	## now reverse order, as p is the loser now
	df_trim=getLoserStats(df)
	opponents.append(df_trim)
	names = pd.concat(opponents)
	return(names)


## Stats list for feature usage: 
	# stats - (first_in/serve_pts)First serve percentage ---Yes, w_1stin/w_svpt - l_1stin/l_svpt
	# stats - (aces) Aces ---Yes (w_ace - _ace)
	# stats - (dfs) Double faults ---Yes
	# stats - (unforced) Unforced errors ---NO
	# stats - (first_won/first_in) Percentage of points won on first serve --Yes
	# stats - (second_won/second_in)Percentage of points won on second serve ---Yes
	# stats - (return_pts_won/return_pts)Percentage of receiving points won ---Yes, player 1 svpt = player 2 return points according to Yi
	# stats - (bk_pts) Break points (won, total) ---Yes
	# netpoints - (pts_won/net_pts)Net approaches (won, total) ---NO
	# stats - (second_won + first_won + return_pts_won/serve_pts + return_pts)Total points won
	# NEW stats - difference in ranking (p - opponent)
	# NEW stats - difference in seed (p - opponent)

## This returns historical averages for player p before match_date, calculated from matches
## matches are calculated from p's matches against his opponents
def ComputeHistoricalAvg(p, match_date, matches):
	## match_date format: yyyymmdd
	historical = matches[matches.tourney_date < match_date]
	grouped = historical.groupby('player_ID')
	res = grouped.aggregate(np.mean)
	return(res)

def ComputeHistoricalAvgAll(matchdate, matches):
	## match_date format: yyyymmdd
	names = pd.DataFrame()
	df = matches
	df = df[df.tourney_date < matchdate]
	df_trim = getWinnerStats(df)
	opponents = list()
	opponents.append(df_trim)
	df = df[df.tourney_date < matchdate]
	df_trim=getLoserStats(df)
	opponents.append(df_trim)
	names = pd.concat(opponents)
	historical = names[names.tourney_date<matchdate]
	res = historical.mean()
	return(res)

def ComputeAvgFromCommonOpp(op1, op2):
	# df = pd.DataFrame({'duration_minutes': op1.duration_minutes - op2.duration_minutes,
	# 					'rank_diff':op1.rank_diff - op2.rank_diff, # we use winner-loser, so likely a negative number
	# 					'rank_pts_diff':op1.rank_pts_diff - op2.rank_pts_diff,
	# 					'height_diff': op1.height_diff - op2.height_diff,
	# 					'first_serve_perc_diff': op1.first_serve_perc_diff - op2.first_serve_perc_diff,
	# 					'ace_diff': op1.ace_diff - op2.ace_diff, 
	# 					'df_diff': op1.df_diff - op2.df_diff, # double faults
	# 					'1st_serve_pts_won_diff': op1['1st_serve_pts_won_diff'] - op2['1st_serve_pts_won_diff']
	# 					# '2nd_serve_pts_won_diff': df.l_2ndWon*1.0/(df.l_svpt - df.l_1stIn) - df.w_2ndWon*1.0/(df.w_svpt - df.w_1stIn),
	# 					# 'bp_faced_diff':df.l_bpFaced - df.w_bpFaced, # break points faced
	# 					# 'bp_saved_diff':df.l_bpSaved - df.w_bpSaved, # break points saved
	# 					# 'bp_saving_perc_diff': df.l_bpSaved*1.0/df.l_bpFaced - df.w_bpSaved*1.0/df.w_bpFaced,
	# 					# 'SvGms_diff':df.l_SvGms - df.w_SvGms, # serve game diff
	# 					# 'svpt_diff': df.l_svpt - df.w_svpt # this tracks return points
	# 					})
	# print(op1,op2)
	return(np.subtract(op1, op2))

# main()
def main():
	# atpmatches = readATPMatches("../tennis_atp")
	atpmatches = readATPMatchesParseTime("../tennis_atp")

	#res = pd.DataFrame()
	test = pd.read_csv('../yi_processing/joined_player_match.csv',
						 index_col=None,
						 header=0,encoding = "ISO-8859-1")
	print(test['match_id'].unique().size)
	results = pd.DataFrame()
	#results.column = ['p1','p2','d','1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 'match_num', 'match_result', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff', 'p1', 'p2', 'd']

	no_match_record_count_p1 = 0
	no_match_record_count_p2 = 0

	no_common_opponent_count =0
	has_common_opponent_count = 0
	res = pd.DataFrame(np.nan,index =xrange(1),columns = xrange(25))
	print(res.shape)
	container = list()
	for i in xrange(test.shape[0]):
		p1 = test.player_1_id[i]
		p2 = test.player_2_id[i]
		d = test.Date[i]
		p1opponents = FindOpponents(p1, atpmatches, d)
		p2opponents = FindOpponents(p2, atpmatches, d)
		(op1, op2) = FindCommonOpponentStats(p1opponents, p2opponents)
		if (op1.shape[0] ==0 or op2.shape[0] == 0):
			## check if we have any common opponent record before the match date, use date constraint
			## if not, use player's historical performance regardless of common opponent
			avgp1 = ComputeHistoricalAvg(p1, d, p1opponents)
			avgp2 = ComputeHistoricalAvg(p2, d, p2opponents)
			if (avgp1.shape[0] == 0 or avgp2.shape[0] ==0):
				res = pd.DataFrame(np.nan,index =xrange(1),columns = xrange(25))
				res['no_historical_data'] = 1
				no_match_record_count_p1+=1
			# 	res = ##preserve column headers, fill with NA
			if (avgp1.shape[0] > 0 and avgp2.shape[0] > 0):
				res = ComputeAvgFromCommonOpp(avgp1, avgp2)
				no_common_opponent_count +=1
				res['no_common_opponent'] = 1
		else:
			## else, compute using common opponent
			avgp1 = ComputeHistoricalAvg(p1, d, op1)
			avgp2 = ComputeHistoricalAvg(p2, d, op2)
			res = ComputeAvgFromCommonOpp(avgp1, avgp2)
			has_common_opponent_count += 1
			res['no_common_opponent'] = 0
			res['no_historical_data'] = 0

		res['player_1_id'] = p1
		res['player_2_id'] = p2
		res['Date'] = d
		#container.append((p1,p2,d,res))
		results = results.append(res)
	print('no common opponent',no_common_opponent_count)
	print('no historical data p1',no_match_record_count_p1)
	print('no historical data',no_match_record_count_p2)

	print('has common opponents',has_common_opponent_count)
		# if i % 15 == 0:
		# 	print(i * 100.0 / test.shape[0], 'percent')
	#results = pd.DataFrame(container)
	#print(results.iloc[0])

	#results.to_csv('common_test.csv', sep=",",quoting=3, encoding = "ISO-8859-1")
	joined_df = pd.merge(test,results,how="left",on=['player_1_id','player_2_id','Date'])
	print('joineddf',joined_df['match_id'].unique().size)
	joined_df.to_csv('joined_hist_match.csv', sep=",",encoding = "ISO-8859-1")
	return

if __name__ == '__main__':
	main()