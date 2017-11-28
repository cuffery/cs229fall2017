import pandas as pd
import numpy as np

def importData():
    match_data = pd.read_csv('../tennis_MatchChartingProject/charting-m-matches.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_data['Date'] = pd.to_datetime(match_data['Date'])
    match_data = match_data.loc[match_data['Date'].dt.year >1999]
    match_data = match_data.loc[match_data['Date'].dt.year <2017]
    match_data['Date']= match_data.Date.map(lambda x: x.strftime('%Y-%m-%d'))
    return match_data

def processMatchDetails(match_data):
    ls = ['match_id','Date','Tournament']

    #set dummy variable for player hand
    match_data['player_1_right'] = np.where(match_data['Pl 1 hand']=='R',1,0)
    match_data['player_2_right'] = np.where(match_data['Pl 2 hand']=='R',1,0)
    ls.extend(('player_1_right','player_2_right'))

    #set dummy variable for court surface
    match_data['surface_hard'] = np.where(match_data['Surface']=='Hard',1,0)
    match_data['surface_grass'] = np.where(match_data['Surface']=='Grass',1,0)
    match_data['surface_clay'] = np.where(match_data['Surface']=='Clay',1,0)
    ls.extend(('surface_hard','surface_grass','surface_clay'))

    # get names and rename cols
    match_data['p1_name'] = match_data['Player 1'].str.strip()
    match_data['p2_name'] = match_data['Player 2'].str.strip()
    ls.extend(('p1_name','p2_name'))

    # get first name, first initial and last name of players
    match_data['player_1_fname'] = match_data['Player 1'].str.strip().str.split(' ').str[0]
    match_data['player_1_lname'] = match_data['Player 1'].str.strip().str.split(' ').str[-1]
    match_data['player_2_fname'] = match_data['Player 2'].str.strip().str.split(' ').str[0]
    match_data['player_2_lname'] = match_data['Player 2'].str.strip().str.split(' ').str[-1]


    match_data['player_1_finit'] = match_data['player_1_fname'].str[0]
    match_data['player_2_finit'] = match_data['player_2_fname'].str[0]

    ls.extend(('player_1_fname','player_1_lname','player_2_fname','player_2_lname','player_1_finit','player_2_finit'))

    # get grand slam
    # bug here. all grand_slam = 0. TODO: fix
    conditions = [(match_data['Tournament']== 'Wimbledon') | (match_data['Tournament']== 'Australian Open') | (match_data['Tournament']== 'Roland Garros') | (match_data['Tournament']== 'US Open') ]
    match_data['grand_slam'] = np.select(conditions,[0], default = 0)
    ls.append('grand_slam')

    return match_data[ls]

def main():
    raw_match_data = importData()
    processed_match_data = processMatchDetails(raw_match_data)

    processed_match_data.to_csv('processed_match_details.csv', sep=",",quoting=3, encoding = "ISO-8859-1")
    return

if __name__ == '__main__':
    main()