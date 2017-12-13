import pandas as pd
import numpy as np
import datetime
import math

def importData(filename):
    match_data = pd.read_csv(filename, delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
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

    match_data['best_of'] = match_data['Best of']
    ls.append('best_of')

    return match_data[ls]

# util function
def preparse_date(t):

    string = str(ts)
    tsdt = datetime.date(int(string[:4]), int(string[4:6]), int(string[6:]))

    return 


def run(source_dir, out_filename):
    raw_match_data = importData(source_dir+'charting-m-matches.csv')
    processed_match_data = processMatchDetails(raw_match_data)
    processed_match_data.to_csv(out_filename, sep=",",quoting=3, encoding = "ISO-8859-1")

    
    # find the age of each player on the match day: 'p1_mday_age' and 'p2_mday_age',
    # find date of birth from atp_players, calculate age from game data and dob
    player_info = pd.read_csv(source_dir + '../tennis_atp/atp_players.csv',
                         index_col=None,
                         header=0,
                         encoding = "ISO-8859-1")

    match_info = pd.read_csv(source_dir + '../data_scripts/match_info/match_details.csv',
                         index_col=None,
                         header=0,
                         encoding = "ISO-8859-1")

    match_info['Date'] = pd.to_datetime(match_info['Date'])


    player_info.columns = ['player_id','p_fname','p_lname','player_hand','player_dob','player_country']

    player_info['name'] = player_info['p_fname'] + " " + player_info['p_lname']

    # use 27 as place holder, also default value
    match_info['p1_mday_age'] = 27.0
    match_info['p2_mday_age'] = 27.0

    for i in range(match_info.shape[0]):
        p1_name = match_info.p1_name[i]
        p2_name = match_info.p2_name[i]
        match_date = match_info.Date[i]
        match_id = match_info.match_id[i]

        p1_info = player_info[player_info.name == p1_name]
        p2_info = player_info[player_info.name == p2_name]

        if (p1_info.shape[0] == 1 and not math.isnan(p1_info.player_dob.iloc[0])):
            #match_info.set_value(i, 'p1_mday_age', (match_date - pd.to_datetime(str(int(p1_info.player_dob.iloc[0])))).days/365)
            match_info.at[i,'p1_mday_age'] = (match_date - pd.to_datetime(str(int(p1_info.player_dob.iloc[0])))).days/365
        if (p2_info.shape[0] == 1 and not math.isnan(p2_info.player_dob.iloc[0])):
            match_info.at[i,'p2_mday_age'] = (match_date - pd.to_datetime(str(int(p2_info.player_dob.iloc[0])))).days/365

    # find age

    match_info.to_csv(out_filename, sep=",",quoting=3, encoding = "ISO-8859-1")

    return

def main():
    run(source_dir = '../../tennis_MatchChartingProject/', out_filename = 'match_details.csv')
    return

if __name__ == '__main__':
    main()