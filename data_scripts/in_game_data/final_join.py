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
import pandas as pd

def prepareHistData(d_):
    # shape before drop (1426, 67)
    # no common opponent 74
    # no historical data p1 or p2 84
    # has common opponents 1269
    col = ['match_id', 'Tournament', 'surface_hard', 'surface_grass', 'surface_clay', #grand_slam TODO: enable when bug fixed
           '1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff', 'bp_faced_diff',
           'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'duration_minutes', 'first_serve_perc_diff',
           'height_diff', 'match_num', 'opponent_ID', 'rank_diff', 'rank_pts_diff', #'match_result' do we need this here?
           'same_handedness', 'svpt_diff']

    d = d_[col].dropna(axis=0, how='any')
    #print(d.shape) # (1012, 22), i.e. 414 rows are dropped. we only expect 84 rows to be dropped.
    # TODO: eddie/yi: investigate and fix the unexpected NAN s in joined_hist_match.csv
    # to see dropped rows: r = d_[col][d_[col].isnull().any(axis=1)]

    # make a copy of the data, negate the common oppo stats, and set player to 2
    d2 = d.copy()
    # TODO: verify correctness of the cols:
    common_oppo_stats =['1st_serve_pts_won_diff', '2nd_serve_pts_won_diff', 'SvGms_diff', 'ace_diff',
            'bp_faced_diff', 'bp_saved_diff', 'bp_saving_perc_diff', 'df_diff', 'first_serve_perc_diff',
            'height_diff', 'rank_diff', 'rank_pts_diff', 'svpt_diff']

    d2[common_oppo_stats] = d2[common_oppo_stats] * -1

    # set player in d to 1 (because common opp data are from p1's perspective), in d2 to 2
    d['player'] = 1
    d2['player'] = 2

    r = pd.concat([d, d2])
    return r

def main():
    hist_data_ = pd.read_csv('../ec_processing/joined_hist_match.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    # drop null and separate data for p1 and p2
    hist_data = prepareHistData(hist_data_)

    # inner join with curr_match_data
    curr_match_data = pd.read_csv('../yubo_processing/current_match_data.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

    joined_data = pd.merge(hist_data, curr_match_data, how="inner",on=['match_id','player'])
    joined_data.to_csv('../final_joined_data.csv', sep=",",quoting=3, encoding = "ISO-8859-1")
    # print(joined_data.shape) = (5224, 35)
    # TODO: investigate the data cleaning process. 5224 is quite low. 

    return

if __name__ == '__main__':
    main()
