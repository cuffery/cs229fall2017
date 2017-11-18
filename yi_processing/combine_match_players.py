import pandas as pd
import numpy as np
from datetime import datetime

# function for parsing strings using specific format

def importProcessed():
    player_ranking = pd.read_csv('../yi_processing/player_ranking.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_details = pd.read_csv('../yi_processing/processed_match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    player_ranking['ranking_date'] = pd.to_datetime(player_ranking['ranking_date'])
    match_details['Date'] = pd.to_datetime(match_details['Date'])
    
    return player_ranking,match_details

def getClosestDate(matchdate,player_rank_history_date):
    return np.amax(player_rank_history_date[player_rank_history_date<matchdate])

def getPlayer()

def mergeMatchRanking(player_ranking,match_details):
    #1. find match with first init and last name

    #2. if more than 1 distinct, then match first name and last name

    #3. find closest date


    matched_ranking_date = player_ranking[player_ranking['player_lname']=='Federer']['ranking_date']

    print(getClosestDate(match_details[match_details['player_2_lname']=='Federer']['Date'].iloc[0],matched_ranking_date))


    return

def main():
    player_ranking, match_details = importProcessed()
    mergeMatchRanking(player_ranking,match_details)
    return

if __name__ == '__main__':
    main()