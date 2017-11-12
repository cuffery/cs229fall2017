import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd

#match_data = pd.read_csv('charting-m-matches.csv')
match_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-matches.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
point_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-points.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

match_data.loc[match_data['Pl 1 hand']=='R','player_1_right']=1
match_data.loc[match_data['Pl 2 hand']=='R','player_2_right']=1

match_data.loc[match_data['Surface']=='Hard','surface_hard']=1
match_data.loc[match_data['Surface']=='Grass','surface_grass']=1
match_data.loc[match_data['Surface']=='Clay','surface_clay']=1




print(match_data.iloc[0])
print(match_data.iloc[1])

df = pd.DataFrame()
df['match_id'] = match_data['match_id']
df['player_1'] = match_data['Player 1']
df['player_2'] = match_data['Player 2']

df['player_1_right'] = match_data['player_1_right']
df['player_2_right'] = match_data['player_2_right']
df['tournament'] = match_data['Tournament']




print(df)




#print(train)

#match_data = genfromtxt('charting-m-matches.csv', delimiter=',')
#print(match_data)