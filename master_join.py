import numpy as np
import pandas as pd

def importAllData():
    joined_hist_match = pd.read_csv('./ec_processing/joined_hist_match.csv',
                        index_col=None,
                        header=0,encoding = "ISO-8859-1")

    atp_label = pd.read_csv('./ec_processing/match_label_ATP.csv',
                            index_col=None,
                            header=0,encoding = "ISO-8859-1")

    curr_match_data = pd.read_csv('./yubo_processing/current_match_data.csv',
                            index_col=None,
                            header=0,encoding = "ISO-8859-1")
    return joined_hist_match,atp_label,curr_match_data

def getP1(joined_hist_match,atp_label,curr_match_data):
    df = pd.DataFrame()
    print('joined hist',joined_hist_match.shape[0])
    print('atp label',atp_label.shape[0])
    p1_curr = curr_match_data[curr_match_data['player'] == 1]
    p1_match_col = ['match_id','player_1_id','player_1_fname','player_1_lname','player_1_right','Date']
    p1_match = joined_hist_match[p1_match_col]
    print('p1 match',p1_match.shape[0])
    #print(p1_match.iloc[0])
    p1_match = p1_match.merge(atp_label[['match_result','player_ID','tourney_date']],how ='left',right_on=['player_ID','tourney_date'],left_on=['player_1_id','Date'],validate = 'one_to_one')
    p1_match['player'] = 1
    print(p1_match.shape[0])
    #print(p1_match.shape[0])
    p1_match.columns = ['match_id','player_id','player_fname','player_lname','player_right','Date','match_result','player_ID','tourney_date','player']
    #print(p1_match.iloc[0])
    df = pd.merge(p1_curr,p1_match,how='left',on=['match_id','player'])
    #print(df.iloc[0])
    #join p1 match result
    #p1_match_result = pd.DataFrame()
    return df

def getMatchDetails(joined_hist_match):
    match_col = ['match_id','surface_hard','surface_grass','surface_clay','Date','grand_slam']
    return joined_hist_match[match_col]

def main():
    joined_hist_match,atp_label,curr_match_data = importAllData()
    #print(list(joined_hist_match))
    #print(list(atp_label))
    #print(list(curr_match_data))
    #getP1(joined_hist_match,atp_label,curr_match_data)
    print(joined_hist_match['match_id'].unique().size)
    print(joined_hist_match['match_id'].unique().size)

    return


if __name__ == '__main__':
	main()