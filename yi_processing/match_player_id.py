import pandas as pd
import numpy as np
import glob
from datetime import datetime
import match_details


def readATPMatchesParseTime(dirname):
    allFiles = glob.glob(dirname + "/atp_matches_" + "20??.csv")[:-1] # -1 to avoid 2017 since its incomplete
    matches = pd.concat((pd.read_csv(file,encoding = "ISO-8859-1") for file in allFiles ))
    return matches


def prepareMatchData(data_):
    # drop 1 dup
    match_details = data_.drop_duplicates(['match_id'])

    match_details = match_details[['Date','match_id','player_1_fname','player_1_lname','player_2_fname','player_2_lname']]

    match_details['Date'] = match_details['Date'].apply(pd.to_datetime, format = '%Y-%m-%d')

    return match_details


def prepareATPData():
    atp_matches = readATPMatchesParseTime("../tennis_atp")

    #get first name, first initial and last name of players
    atp_matches['winner_fname'] = atp_matches['winner_name'].str.strip().str.split(' ').str[0]
    atp_matches['winner_lname'] = atp_matches['winner_name'].str.strip().str.split(' ').str[-1]
    atp_matches['loser_fname'] = atp_matches['loser_name'].str.strip().str.split(' ').str[0]
    atp_matches['loser_lname'] = atp_matches['loser_name'].str.strip().str.split(' ').str[-1]

    atp_matches['tourney_date'] = atp_matches['tourney_date'].apply(pd.to_datetime, format = '%Y%m%d')

    return atp_matches[['tourney_date','winner_fname','loser_fname','winner_lname','loser_lname','winner_id','loser_id']]


def findPlayerIdForMatches(match_data, atp_data):
    '''
    dup = match_data[match_data.duplicated(['player_1_lname','player_1_fname','Date'],keep = False)]
    dup.to_csv('dup.csv')
    print(dup[['match_id']])
    print(match_data[['player_1_lname','player_1_fname','Date']].shape[0])
    '''
    dup = atp_data[atp_data.duplicated(['winner_lname','loser_lname','tourney_date'],keep = False)]
    dup.to_csv('dup.csv')

    join_winner_1_loser_2 = pd.merge(match_data,atp_data[['tourney_date','loser_fname','loser_lname','winner_fname','winner_lname','winner_id','loser_id']],left_on=['Date','player_1_lname','player_2_lname'],right_on = ['tourney_date','winner_lname','loser_lname'],how = 'inner')
    join_winner_2_loser_1 = pd.merge(match_data,atp_data[['tourney_date','loser_fname','loser_lname','winner_fname','winner_lname','winner_id','loser_id']],left_on=['Date','player_1_lname','player_2_lname'],right_on = ['tourney_date','loser_lname','winner_lname'],how = 'inner')
    #dup = match_data[match_data.duplicated(['player_1_lname','player_2_lname','Date'],keep = False)]
    #dup.to_csv('dup.csv')


    print('match data',match_data.shape[0])
    print('player_1 = winner',join_winner_1_loser_2.shape[0])
    print('player_2 = winner',join_winner_2_loser_1.shape[0])
    return

def main():
    # read match data for matches and atp match data for player ids
    match_details.main()
    raw_match_details = pd.read_csv('../yi_processing/processed_match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    match_data = prepareMatchData(raw_match_details)

    atp_match_data = prepareATPData()

    # join
    result = findPlayerIdForMatches(match_data, atp_match_data)

    #print(result.shape)

    return


if __name__ == '__main__':
    main()