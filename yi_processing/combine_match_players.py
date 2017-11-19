import pandas as pd
import numpy as np
from datetime import datetime
import player_details
import match_details


# function for parsing strings using specific format
def importProcessed():
    player_details = pd.read_csv('../yi_processing/player_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_details = pd.read_csv('../yi_processing/processed_match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    #player_details = pd    .read_csv('../yi_processing/player_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_details['Date'] = pd.to_datetime(match_details['Date'])
    
    return player_details,match_details
'''
def getClosestDate(matchdate,player_rank_history_date):
    return np.amax(player_rank_history_date[player_rank_history_date<matchdate])
'''
def getPlayerIDTable(player_details,match_details):
    match_player1_info = ['player_1_fname','player_1_lname','player_1_finit']
    match_player2_info = ['player_2_fname','player_2_lname','player_2_finit']
    player_info = ['player_fname','player_lname','player_finit','player_id']

    player_from_details = player_details[player_info].drop_duplicates()
    player1_names_from_match = match_details[match_player1_info].drop_duplicates()
    player2_names_from_match = match_details[match_player2_info].drop_duplicates()
    player1_names_from_match.columns = ['player_fname','player_lname','player_finit']
    player2_names_from_match.columns =  ['player_fname','player_lname','player_finit']
    
    player_from_match = pd.concat([player1_names_from_match,player2_names_from_match])
    player_from_match = player_from_match.drop_duplicates()
    
    player_from_match = player_from_match[pd.notnull(player_from_match['player_lname'])]
    

    df = pd.merge(player_from_match,player_from_details, on = ['player_lname','player_fname','player_finit'],how = 'left')#,validate = 'one_to_many')

    df = df[pd.notnull(df['player_id'])]
    df['player_concat'] = df['player_fname'] + df['player_lname'] + df['player_finit']
    return df[['player_concat','player_id']] 

def getPlayerID(player_id_table,match_details):   
    match_details['player1_concat'] = match_details['player_1_fname'] + match_details['player_1_lname'] + match_details['player_1_finit']
    match_details['player2_concat'] = match_details['player_2_fname'] + match_details['player_2_lname'] + match_details['player_2_finit']

    match_details = match_details.merge(player_id_table,left_on='player1_concat',right_on = 'player_concat',how = 'left')
    match_details = match_details.rename(columns = {'player_id':'player_1_id'})
    
    match_details = match_details.merge(player_id_table,left_on='player2_concat',right_on = 'player_concat',how = 'left')
    match_details = match_details.rename(columns = {'player_id':'player_2_id'})

    match_details =  match_details.drop(['player_concat_x','player_concat_y','player1_concat','player2_concat'],axis = 1)
    return match_details

def mergeMatchRanking(player_details,match_details):
    player_id_table = getPlayerIDTable(player_details,match_details)

    match_details = getPlayerID(player_id_table,match_details)


    return match_details

def main():
    player_details.main()
    match_details.main()
    player_details_data, match_details_data = importProcessed()
    match_details_data = mergeMatchRanking(player_details_data,match_details_data)
    match_details_data.to_csv('joined_player_match.csv', sep=",",quoting=3, encoding = "ISO-8859-1")

    return

if __name__ == '__main__':
    main()