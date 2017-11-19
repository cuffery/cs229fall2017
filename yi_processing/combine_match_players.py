import pandas as pd
import numpy as np
from datetime import datetime

# function for parsing strings using specific format

def importProcessed():
    player_ranking = pd.read_csv('../yi_processing/player_ranking.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_details = pd.read_csv('../yi_processing/processed_match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    #player_details = pd.read_csv('../yi_processing/player_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    player_ranking['ranking_date'] = pd.to_datetime(player_ranking['ranking_date'])
    match_details['Date'] = pd.to_datetime(match_details['Date'])
    
    return player_ranking,match_details

def getClosestDate(matchdate,player_rank_history_date):
    return np.amax(player_rank_history_date[player_rank_history_date<matchdate])

def getPlayer(player_from_ranking,player_from_match):
    #takes player_details (array) from player table, and one player from each match
    player_from_match.columns = ['player_fname','player_lname','player_finit']
    
    print("match table",player_from_match.shape)
    print("player table",player_from_ranking.shape)

    df = pd.merge(player_from_match,player_from_ranking, on = ['player_fname','player_lname'],how = 'left')
    
    print("merged table",df.shape)
    
    print("where merged table is null",df[pd.isnull(df['player_id'])])
    #print("where player table is null",player_from_ranking[pd.isnull(player_from_ranking)])
    #print("where match table is null",player_from_match[pd.isnull(player_from_match)])


    #return player name in playing ranking table
    return

def mergeMatchRanking(player_ranking,match_details):
    match_player1_info = ['player_1_fname','player_1_lname','player_1_finit']
    match_player2_info = ['player_2_fname','player_2_lname','player_2_finit']
    player_info = ['player_fname','player_lname','player_finit','player_id']

    player_names_from_ranking = player_ranking[player_info].drop_duplicates()

    getPlayer(player_names_from_ranking,match_details[match_player1_info])


    #getPlayer(player_ranking[player_info],match_details[match_player1_info].iloc[0])


    #1. find match with first init and last name

    #2. if more than 1 distinct, then match first name and last name

    #3. find closest date
    #matched_ranking_date = player_ranking[player_ranking['player_lname']=='Federer']['ranking_date']

    #print(getClosestDate(match_details[match_details['player_2_lname']=='Federer']['Date'].iloc[0],matched_ranking_date))

    return

def main():
    player_ranking, match_details = importProcessed()
    mergeMatchRanking(player_ranking,match_details)
    return

if __name__ == '__main__':
    main()