import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd

#match_data = pd.read_csv('charting-m-matches.csv')
match_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-matches.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
point_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-points.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
player_data = pd.read_csv('../tennis_atp/atp_players.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

player_data.columns = ['player_id','player_firstname','player_lastname','player_hand','player_dob','player_nat']

player_data['player_firstinit'] = player_data['player_firstname'].str[0]

match_data.loc[match_data['Pl 1 hand']=='R','player_1_right']=1
match_data.loc[match_data['Pl 2 hand']=='R','player_2_right']=1

match_data.loc[match_data['Surface']=='Hard','surface_hard']=1
match_data.loc[match_data['Surface']=='Grass','surface_grass']=1
match_data.loc[match_data['Surface']=='Clay','surface_clay']=1

print(match_data.iloc[0])
print(point_data.iloc[0])
print(player_data.iloc[0])

df1 = pd.DataFrame()

df1['id'] = match_data['match_id'] + '_1'

#player information
df1['player'] = match_data['Player 1']
df1['player_firstname'] = df1['player'].str.split(' ').str[0]
df1['player_lastname'] = df1['player'].str.split(' ').str[-1]
df1['player_firstinit'] = df1['player_firstname'].str[0]
df1['player_right'] = match_data['player_1_right']

#player information from player_data table

#match information
df1['tournament'] = match_data['Tournament']
df1['is_clay'] = match_data['surface_clay']
df1['is_grass'] = match_data['surface_grass']
df1['is_hard'] = match_data['surface_hard']
df1['date'] = match_data['Date']


print(df1[df1['player_lastname'].isin(player_data['player_lastname'])] )
df2.user_id.isin(df1.user_id)

#def getPlayer(x,player):
    #player[(x['player_lastname'] == player['player_lastname']) & (x['player_firstinit'] == player['player_firstinit'])]

print(df1['player_lastname'][0])
print(df1['player_lastname'].dtype)

'''
df['match_id'] = match_data['match_id']
df['player_1'] = match_data['Player 1']
df['player_2'] = match_data['Player 2']

df['player_2_right'] = match_data['player_2_right']
'''






#print(df1)




#print(train)

#match_data = genfromtxt('charting-m-matches.csv', delimiter=',')
#print(match_data)