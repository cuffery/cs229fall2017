#   GenerateATPlabel.py
#   
#   Script that generates match results from ATP matches
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
	df = CPS.readATPMatchesParseTime("../tennis_atp")
	df_win = pd.DataFrame({'tourney_id':df.tourney_id,
							'tourney_name':df.tourney_name, 
							'surface':df.surface, 
							'tourney_date':df.tourney_date, 
							'match_num':df.match_num, 
							'duration_minutes': df.minutes,
							'match_result':1.,
							'player_name':df.winner_name,
							'player_ID':df.winner_id
							})
	df_loss = pd.DataFrame({'tourney_id':df.tourney_id,
							'tourney_name':df.tourney_name, 
							'surface':df.surface, 
							'tourney_date':df.tourney_date, 
							'match_num':df.match_num, 
							'duration_minutes': df.minutes,
							'match_result':0.,
							'player_name':df.loser_name,
							'player_ID':df.loser_id
							})
	frames = [df_win, df_loss]
	result = pd.concat(frames)
	# print(result.dtypes)
	result.to_csv('match_label_ATP.csv', sep=",",quoting=3, encoding = "ISO-8859-1")

	return

if __name__ == '__main__':
	main()