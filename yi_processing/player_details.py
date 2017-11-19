import pandas as pd
import numpy as np


def importPlayerData():
    player = pd.read_csv('../tennis_atp/atp_players.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1",dtype = object)
    player.columns = ['player_id','player_fname','player_lname','player_hand','player_dob','player_country']
    player['player_birthyear'] = player.player_dob.str[0:4]
    player['player_id'] = player['player_id'].astype(np.int64)
    player['player_finit'] = player['player_fname'].str[0]
    
    player['player_lname'] = player['player_lname'].str.split(' ').str[-1]
    player['player_fname'] = player['player_fname'].str.split(' ').str[0]

    player = player.drop('player_dob',1)
    return player

def importRankingData():
    ranking_00 = pd.read_csv('../tennis_atp/atp_rankings_00s.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1",low_memory = False)
    ranking_10 = pd.read_csv('../tennis_atp/atp_rankings_10s.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1",low_memory = False)
    ranking_00.columns = ['ranking_date','rank','player_id','ranking_points']
    ranking_10.columns = ['ranking_date','rank','player_id','ranking_points']
    ranking = pd.concat([ranking_00,ranking_10])
    
    ranking['ranking_date'] = pd.to_datetime(ranking['ranking_date'],format = '%Y%m%d')
    ranking['ranking_date']= ranking.ranking_date.map(lambda x: x.strftime('%Y-%m-%d'))
    return ranking

def matchPlayerRank(player,ranking):
    player_rank = pd.merge(ranking, player, on='player_id',how = 'outer')
    return player_rank

def main():
    player = importPlayerData()
    ranking = importRankingData()
    player_ranking = matchPlayerRank(player,ranking)
    player.to_csv('player_details.csv', sep=",",quoting=3, encoding = "ISO-8859-1")
    player_ranking.to_csv('player_ranking.csv', sep=",",quoting=3, encoding = "ISO-8859-1")
    return

if __name__ == '__main__':
    main()