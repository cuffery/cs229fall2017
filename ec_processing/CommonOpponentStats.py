#   CommonOpponentStats.py
#   
#   Script that takes in two players and outputs their stats calcualted through common opponent
#
#   Created by: Eddie Chen
#	Last edit: 11/16/2017 14:07


import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd

## This returns a list of common opponent between p1 and p2
def FindCommonOpponent(p1,p2):

    return

## This returns the stats of player p and his opponents in op (an array)
def CalcStats(p, op):

    return

# main()
def main():
    #match_data = pd.read_csv('charting-m-matches.csv')
	match_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-matches.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
	point_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-points.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

	print(match_data.iloc[0])
	print(match_data.iloc[1])

	df = pd.DataFrame()
	df['match_id'] = match_data['match_id']
	df['player_1'] = match_data['Player 1']
	df['player_2'] = match_data['Player 2']

	print(df)

    return

if __name__ == '__main__':
    main()