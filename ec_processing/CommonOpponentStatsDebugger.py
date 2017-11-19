#   CommonOpponentStatsDebugger.py
#   
#   Script that takes in tests CommonOpponentStats
#
#   Created by: Eddie Chen
#	Last edit: 11/19/2017 21:07

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import glob
import datetime
import CommonOpponentStats as CPS


def main():
	# atpmatches = readATPMatches("../tennis_atp")
	atpmatches = CPS.readATPMatchesParseTime("../tennis_atp")

	## input p1 and p2; test case
	d = '20161101'
	p1 = 105060 ##
	p1opponents = CPS.FindOpponents(p1, atpmatches, d)

	print(p1opponents.shape[0])
	p2 = 104252 ##
	p2opponents = CPS.FindOpponents(p2, atpmatches, d)
	print(p2opponents.shape[0])

	(op1, op2) = CPS.FindCommonOpponentStats(p1opponents, p2opponents)
	print(op1.shape[0])
	print(op2.shape[0])
	avgp1 = CPS.ComputeHistoricalAvg(p1, d, op1)
	avgp2 = CPS.ComputeHistoricalAvg(p2, d, op2)

	res = CPS.ComputeAvgFromCommonOpp(avgp1, avgp2) ## don't use player id column
	print(res)

if __name__ == '__main__':
	main()