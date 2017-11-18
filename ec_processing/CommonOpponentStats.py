#   CommonOpponentStats.py
#   
#   Script that takes in two players and outputs their stats calcualted through common opponent
#
#   Created by: Eddie Chen
#	Last edit: 11/16/2017 14:07

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

## This returns a list of common opponent between p1 and p2
def FindCommonOpponent(p1,p2):

    return

## This returns the stats of player p and his opponents in op (an array)
def CalcStats(p, op):

    return

# main()
def main():
	atpmatches = readATPMatches("../tennis_atp")
	print(atpmatches.shape)

	return

if __name__ == '__main__':
    main()