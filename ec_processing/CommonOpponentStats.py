#   CommonOpponentStats.py
#   
#   Script that takes in two players and outputs their stats calcualted through common opponent
#
#   Created by: Eddie Chen
#	Last edit: 11/18/2017 21:07

## Per-set stats for both players - checklist from atp_matches
# stats - (first_in/serve_pts)First serve percentage ---Yes, w_1stin/w_svpt - l_1stin/l_svpt
# stats - (aces) Aces ---Yes (w_ace - _ace)
# stats - (dfs) Double faults ---Yes
# stats - (unforced) Unforced errors ---NO
# stats - (first_won/first_in) Percentage of points won on first serve --Yes
# stats - (second_won/second_in)Percentage of points won on second serve ---Yes
# stats - (return_pts_won/return_pts)Percentage of receiving points won ---NO
# stats - (bk_pts) Break points (won, total) ---Yes
# netpoints - (pts_won/net_pts)Net approaches (won, total) ---NO
# stats - (second_won + first_won + return_pts_won/serve_pts + return_pts)Total points won
# Fastest serve
# Average first serve speed
# Average second serve speed
# Odds (Marathonbet, Pinnacle)

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
    allFiles = glob.glob(dirname + "/atp_matches_" + "????.csv")
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
def FindCommonOpponents(p1opponents, p2opponents):
	opp = pd.DataFrame()
	container = list()
	for i in p1opponents:
		for j in p2opponents:
			if i == j:
				container.append(i)
				# print(i)
				# print(j)
	#opp = pd.concat(container)
	return(container)

## This calculates differential stats from common opponents
def FindCommonOpponentStats(p1opp, p2opp):

	
    return

## This returns a list of opponents for player p
def FindOpponents(p, matches):
## find games where p won, and then where p lost
	names = pd.DataFrame()
	df = matches[matches.winner_name == p]
	opponents = list()
	opponents.append(df.loser_name)
	df = matches[matches.loser_name == p]
	opponents.append(df.winner_name)
	names = pd.concat(opponents)
 	return(names)

## This returns a list of opponents and their game stats for player p
def FindOpponentStats(p, matches):
## find games where p won, and then where p lost
	stats = pd.DataFrame()
	df = matches[matches.winner_name == p]
	opponents = list()
	opponents.append(df)
	df = matches[matches.loser_name == p]
	opponents.append(df)
	stats = pd.concat(opponents)
 	return(stats)


## This returns the stats of player p and his opponents in op (an array)
## helper function
def CalcStats(p, op):
	##TODO
	# stats - (first_in/serve_pts)First serve percentage ---Yes, w_1stin/w_svpt - l_1stin/l_svpt
	# stats - (aces) Aces ---Yes (w_ace - _ace)
	# stats - (dfs) Double faults ---Yes
	# stats - (unforced) Unforced errors ---NO
	# stats - (first_won/first_in) Percentage of points won on first serve --Yes
	# stats - (second_won/second_in)Percentage of points won on second serve ---Yes
	# stats - (return_pts_won/return_pts)Percentage of receiving points won ---NO
	# stats - (bk_pts) Break points (won, total) ---Yes
	# netpoints - (pts_won/net_pts)Net approaches (won, total) ---NO
	# stats - (second_won + first_won + return_pts_won/serve_pts + return_pts)Total points won
	# NEW stats - difference in ranking (p - opponent)
	# NEW stats - difference in seed (p - opponent)

    return

# main()
def main():
	atpmatches = readATPMatches("../tennis_atp")
	# print(atpmatches.shape) ## 55520 rows, 49 fields
	# print(atpmatches[atpmatches.winner_name == 'Rafael Nadal'])
	
	## input p1 and p2; test case
	p1 = 'Rafael Nadal'
	p2 = 'Roger Federer'
	p1opponents = FindOpponents(p1, atpmatches)
	# print(p1opponents) ## returns a dataframe
	p2opponents = FindOpponents(p2, atpmatches)
	p1Stats = FindOpponentStats(p1, atpmatches)
	p2Stats = FindOpponentStats(p2, atpmatches)

	commonOpp = FindCommonOpponents(p1opponents, p2opponents)
	print(commonOpp) ## WE NEED MATCH ID TOO!!!

	return

if __name__ == '__main__':
    main()