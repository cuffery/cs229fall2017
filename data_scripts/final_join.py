#   final_join.py
#
#   Script that reads two processed .csv files:
#       ../ec_processing/joined_hist_data.csv: contains match info and common opponents info, processed by yi and eddie
#       ./curr_match_data.csv: contains per set match data for both players
#
#   This file process joined_hist_data.csv, then join it with curr_match_data.csv on player(1 or 2) and match_id, and output
#   into final_data.csv
#
#   Owner: Yubo Tian

'''
Features provided by each source file:
Data processing step (1):
 - source: charting-m-matches.csv
    ['match_id', 'Player 1', 'Player 2', 'Pl 1 hand', 'Pl 2 hand', 'Gender', 'Date', 'Tournament',
     'Round', 'Time', 'Court', 'Surface', 'Umpire', 'Best of', 'Final TB?', 'Charted by']

 - output (as of Dec 10):
    ['Unnamed: 0', 'match_id', 'Date', 'Tournament', 'player_1_right', 'player_2_right', 'surface_hard',
     'surface_grass', 'surface_clay', 'p1_name', 'p2_name', 'player_1_fname', 'player_1_lname', 
     'player_2_fname', 'player_2_lname', 'player_1_finit', 'player_2_finit', 'grand_slam']

 # TODO: 'p1_game_age', 'p2_game_age', 'best_of'

Data processing step (2):
 - source: tennis_atp/atp_matches_20xx.csv
    ['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num',
     'winner_id', 'winner_seed', 'winner_entry', 'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc',
     'winner_age', 'winner_rank', 'winner_rank_points', 'loser_id', 'loser_seed', 'loser_entry', 
     'loser_name', 'loser_hand', 'loser_ht', 'loser_ioc', 'loser_age', 'loser_rank', 'loser_rank_points',
     'score', 'best_of', 'round', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
     'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon',
     'l_SvGms', 'l_bpSaved', 'l_bpFaced']

 - output (as of Dec 10):
    ['match_id', 'Tournament', 'surface_hard', 'surface_grass', 'surface_clay', '1st_serve_pts_won_diff',
     '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff', 'bp_saved_diff', 
     'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff', 'height_diff', 
     'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', 'same_handedness', 'svpt_diff', 'player']

 # TODO: 'avg_age_lost_to', 'avg_age_win_against'
         'avg_minutes_lost', 'avg_minutes_won'

Data processing step (3):
 - source: charting-m-stats-Overview.csv:
    ['match_id', 'player', 'set', 'serve_pts', 'aces', 'dfs', 'first_in', 'first_won',
     'second_in', 'second_won', 'bk_pts', 'bp_saved', 'return_pts', 'return_pts_won', 
     'winners', 'winners_fh', 'winners_bh', 'unforced', 'unforced_fh', 'unforced_bh']

 - output (as of Dec 10):
    ['Unnamed: 0', 'match_id', 'after_set', 'player', 'aces', 'dfs', 'unforced', '1st_srv_pct',
     'bk_pts', 'winners', 'pts_1st_srv_pct', 'pts_2nd_srv_pct', 'rcv_pts_pct', 'ttl_pts_won']
'''
import pandas as pd
import sys


def prepareHistData(d_):
    # shape before drop (1426, 67)
    # no common opponent 74
    # no historical data p1 or p2 84
    # has common opponents 1269
    # print(list(d_))
    col = ['match_id', 'Tournament', 'surface_hard', 'surface_grass', 'surface_clay', #grand_slam TODO: enable when bug fixed
           '1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff',
           'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'first_serve_perc_diff', 'duration_minutes',
           'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', #'match_result' do we need this here?
           'same_handedness', 'svpt_diff', 'best_of', 'p1_mday_age', 'p2_mday_age']

    # TODO: separate_out p1_avg_minutes_lost, p1_avg_minutes_won, p2_avg_minutes_lost, p2_avg_minutes_won 
    col_p1 = col + ['p1_avg_minutes_lost','p1_avg_minutes_won', 'p1_avg_age_lost_to', 'p1_avg_age_won_against']
    col_p2 = col + ['p2_avg_minutes_lost', 'p2_avg_minutes_won', 'p2_avg_age_lost_to', 'p2_avg_age_won_against']

    d1 = d_.copy()
    d2 = d_.copy()

    d1 = d1[col_p1].dropna(axis=0, how='any')
    #print(d.shape) # (1012, 22), i.e. 414 rows are dropped. we only expect 84 rows to be dropped.
    # TODO: eddie/yi: investigate and fix the unexpected NAN s in joined_hist_match.csv
    # to see dropped rows: r = d_[col][d_[col].isnull().any(axis=1)]

    # make a copy of the data, negate the common oppo stats, and set player to 2


    d2 = d2[col_p2].dropna(axis=0, how='any')

    # TODO: verify correctness of the cols:
    common_oppo_stats =['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff',
            'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'first_serve_perc_diff',
            'height_diff', 'rank_diff', 'rank_pts_diff', 'svpt_diff']

    d2[common_oppo_stats] = d2[common_oppo_stats] * -1

    # set player in d to 1 (because common opp data are from p1's perspective), in d2 to 2
    d1['player'] = 1
    d2['player'] = 2

    #d1.rename({"p1_avg_minutes_lost":"avg_minutes_lost", "p1_avg_minutes_won":"avg_minutes_won"}, axis='columns')
    #d2.rename({"p2_avg_minutes_lost":"avg_minutes_lost", "p2_avg_minutes_won":"avg_minutes_won"}, axis='columns')

    d1 = d1.rename(index=str, columns = {"p1_avg_minutes_lost":"avg_minutes_lost", "p1_avg_minutes_won":"avg_minutes_won",
                                         "p1_avg_age_lost_to":"avg_age_diff_lost_to", "p1_avg_age_won_against":"avg_age_diff_won_against",
                                         "p1_mday_age": "match_day_age", "p2_mday_age": "opponent_match_day_age"})
    d2 = d2.rename(index=str, columns = {"p2_avg_minutes_lost":"avg_minutes_lost", "p2_avg_minutes_won":"avg_minutes_won",
                                         "p2_avg_age_lost_to":"avg_age_diff_lost_to", "p2_avg_age_won_against":"avg_age_diff_won_against",
                                         "p2_mday_age": "match_day_age", "p1_mday_age": "opponent_match_day_age"})

    r = pd.concat([d1, d2])

    #print("prepareHistData",list(r))

    return r


def processJoinedData(joined_data):

    joined_data['match_day_age_diff'] = joined_data['match_day_age'] - joined_data['opponent_match_day_age']

    joined_data['duration_win_loss_rate'] = (joined_data['avg_minutes_won'] - joined_data['avg_minutes_lost']) * 10.0 / joined_data['rally_sum']

    joined_data['age_diff_win_loss_rate'] = (joined_data['avg_age_diff_won_against'] - joined_data['avg_age_diff_lost_to']) / joined_data['match_day_age_diff']

    return joined_data


def main():
    # Overview:
    # (1) processed_match_details.csv contains general match information, and is generated by
    # match_info/match_details.py
    # (2) joined_hist_data.csv is generated by past_perf/AddingCommonOpponnentStats_by_name.py, 
    # through adding the common opponents info to processed_match_details.csv
    # (3) current_match_data.csv contains real time game information, on per set basis. It 
    # is generated by in_game_data/curr_match_data

    # (1)
    print('(1/5) Processing match general information data...')
    sys.path.insert(0, './match_info/') # not a very pythonic approach, but its good enough
    import match_details
    match_details.run(source_dir = '../tennis_MatchChartingProject/', out_filename = './match_info/match_details.csv')
    print('Done.')

    # (2)
    print('(2/5) Generating historical common opponents data for matches. This may take a while...')
    sys.path.insert(0, './past_perf/')
    import AddingCommonOpponentStats_by_name as common_oppo_data
    #common_oppo_data.run(source_dir = '../', out_filename = './past_perf/joined_hist_match.csv')
    print ('\nDone.')

    # (3)
    print('(3/5) Processing real time match data...')
    sys.path.insert(0, './in_game_data/')
    import curr_match_data
    #curr_match_data.run(source_dir = '../tennis_MatchChartingProject/', out_filename = './in_game_data/curr_match_data.csv')
    print('\nDone.')

    # Now Process the labels
    print('(4/5) Extracting labels - result per match and result per set')
    import label_match, label_set
    label_match.run(source_dir = '../tennis_MatchChartingProject/', out_filename = './result_per_match.csv')
    #label_set.run(source_dir = '../tennis_MatchChartingProject/', out_filename = './result_per_set.csv')
    print("\nDone. Output file: ./result_per_match.csv, ./result_per_set.csv")

    # Now perform the final join
    print('(5/5) Final cleaning and join data...')

    hist_data_ = pd.read_csv('./past_perf/joined_hist_match.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    # drop null and separate data for p1 and p2
    hist_data = prepareHistData(hist_data_)

    # inner join with curr_match_data
    curr_match_data_ = pd.read_csv('./in_game_data/curr_match_data.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")


    # join with 'result_per_set.csv' to get the rally_count feature calculated from 'charting-m-points.csv'
    #print('curr_match_data_.shape', curr_match_data_.shape)

    result_per_set = pd.read_csv('./result_per_set.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    rally_sums = result_per_set[['match_id', 'set', 'rally_sum']]

    rally_sums = rally_sums.rename(index=str, columns = {"set":"after_set"}) 

    curr_match_data = pd.merge(curr_match_data_, rally_sums, how="left",on=['match_id','after_set'])
    #print('curr_match_data.shape', curr_match_data.shape)

#    temp = pd.read_csv('./match_info/match_details.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
#    print('1. temp', list( temp ))
#    print('2. hist_data', list( hist_data ))
#    print('3. curr_match_data', list( curr_match_data ))

    joined_data = pd.merge(hist_data, curr_match_data, how="inner",on=['match_id','player'])

    # final process data. because these features are only available after in-game-data is merge with historical-data
    final_joined_data = processJoinedData(joined_data)


    final_joined_data.to_csv('./final_joined_data.csv', sep=",",quoting=3, encoding = "ISO-8859-1")

    print("Done. Output file: ./final_joined_data.csv")
    print('joined_data.shape ' , final_joined_data.shape)# = (5224, 35)
    print(list(final_joined_data))

    return

if __name__ == '__main__':
    main()
