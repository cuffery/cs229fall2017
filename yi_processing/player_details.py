import pandas as pd
import numpy as np

def importPlayerData():
    player_data = pd.read_csv('../tennis_atp/atp_players.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    player_data.columns = ['player_id','player_fname','player_lname','player_hand','player_dob','player_country']
    return player_data

def importRankingData():
    ranking_00 = pd.read_csv('../tennis_atp/atp_rankings_00s.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1",dtype = np.int64)
    ranking_10 = pd.read_csv('../tennis_atp/atp_rankings_10s.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1",dtype = np.int64)
    print(ranking_00.shape)
    print(ranking_10.shape)

    ranking = pd.concat([ranking_00,ranking_10])
    print(ranking.shape)
    return ranking

def main():
    importRankingData()
    return

if __name__ == '__main__':
    main()