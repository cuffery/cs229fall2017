#   curr_match_data.py
#   
#   Script that reads match data and generates .csv for current match status.
#
#   Owner: Yubo Tian
import pandas as pd
import numpy as np
import time
import sys

def processMatch(match):
    r = pd.DataFrame();

    for i in range(int(match['set'].max())+1):
        mset = match[match['set'] <= str(i)]
        g = mset.groupby('player')

        for player, d in g:
            p = pd.DataFrame({
                'match_id' : match.match_id.unique(),
                'player': player,
                'after_set': i,
                'aces': d['aces'].sum(),
                'dfs': d['dfs'].sum(),
                'unforced': d['unforced'].sum(),
                '1st_srv_pct': d['first_in'].sum() / d['serve_pts'].sum(),
                'bk_pts': d['bk_pts'].sum(),
                'winners': d['winners'].sum(),
                'pts_1st_srv_pct': d['first_won'].sum() / d['first_in'].sum(),
                'pts_2nd_srv_pct': 0 if d['second_in'].sum() == 0 else d['second_won'].sum() / d['second_in'].sum(),
                'rcv_pts_pct': d['return_pts_won'].sum() / d['return_pts'].sum(),
                'ttl_pts_won': (d['second_won'].sum()+d['first_won'].sum()+d['return_pts_won'].sum()) / (d['serve_pts'].sum()+d['return_pts'].sum())
            })

            r = r.append(p)
    
    return r

# aggregateOnSet(data)
def aggregateOnSet(data_):
    # rm games before 2000 and rows with set == 'Total'
    data = data_[(data_['match_id'].map(lambda x: x.startswith('20'))) & (data_['set'] != 'Total')]

    # initialize container for processed data
    df = pd.DataFrame(columns=['match_id','after_set','player','aces','dfs','unforced','1st_srv_pct','bk_pts','winners','pts_1st_srv_pct','pts_2nd_srv_pct','rcv_pts_pct','ttl_pts_won'])

    # stats - 'feature name': (calculation from orig data) description
    # stats - 'aces': Aces = (aces)
    # stats - 'dfs': Double faults
    # stats - 'unforced': Unforced errors
    # stats - '1st_srv_pct': (first_in/serve_pts) First serve percentage 
    # stats - 'bk_pts' = Break points (won, total)
    # stats - 'winners' = Winners
    # stats - 'pts_1st_srv_pct' = (first_won/first_in) Percentage of points won on first serve
    # stats - 'pts_2nd_srv_pct' = (second_won/second_in)Percentage of points won on second serve
    # stats - 'rcv_pts_pct' = (return_pts_won/return_pts)Percentage of receiving points won
    # stats - 'ttl_pts_won' = (second_won + first_won + return_pts_won/serve_pts + return_pts) Total points won
    ## TODO stats - 'net_pts' = (pts_won/net_pts)Net approaches (won, total)
   
    grouped = data.groupby('match_id')

    # progress bar 
    sys.stdout.write("[%s]" % (" " * int(len(grouped) / 22)))
    sys.stdout.flush()
    sys.stdout.write("\b" * int(len(grouped) / 22 + 1))
    i = 0
    for match_id, group in grouped:
        i+=1
        m = processMatch(grouped.get_group(match_id))
        df = df.append(m)

        # progress bar
        if ( i % 22 == 0 ):
            sys.stdout.write("=")
            sys.stdout.flush()
    return df


# readFile(filename)
def readFile(filename):
    rawData = pd.read_csv(filename, delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    return rawData;


# debug(data)
# description: container fucntion for testing/ sample code
def debug(data_):
    #print(data.iloc[0])
    # total num of matches
    # print(rawData.match_id.unique().size)
    
    # num of matches in or after year 2000
    # criteria = rawData['match_id'].map(lambda x: x.startswith('20'))
    # data = data_[(data_['match_id'].map(lambda x: x.startswith('20') and not x.startswith('2017')))]
#    data = data_[(data_['match_id'].map(lambda x: x.startswith('20'))) & (data_['set'] != 'Total')]

    # matches2017 = rawData[criteria]
    # print(data.match_id.unique().size)


    # debug: verify the common match_ids from match_details(charting-matches) and curr_match_data(charting-overview)
    curr_match_data = readFile('current_match_data.csv')
    curr_match_id_count = curr_match_data.match_id.unique().size

    match_details_data = readFile('../yi_processing/processed_match_details.csv')
    match_details_match_id_count = match_details_data.match_id.unique().size

    print("debug verification: ", curr_match_id_count, match_details_match_id_count)

    c = curr_match_data[['match_id']].drop_duplicates()
    d = match_details_data[['match_id','Date']]

    s1 = pd.merge(c, d, how='inner', on=['match_id'])

    print("verification: joined details/curr match: ", s1.shape)

    return


# main()
def run(source_dir, out_filename):
    rawData = readFile(source_dir + 'charting-m-stats-Overview.csv')
    #debug(rawData)
    start_time = time.time()
    result = aggregateOnSet(rawData)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    col = ['match_id','after_set','player','aces','dfs','unforced','1st_srv_pct','bk_pts','winners','pts_1st_srv_pct','pts_2nd_srv_pct','rcv_pts_pct','ttl_pts_won']
    result.to_csv(out_filename, sep=",",quoting=3, encoding = "ISO-8859-1", columns=col)
    return


def main():
    run('../../tennis_MatchChartingProject', 'curr_match_data.csv')
    return


if __name__ == '__main__':
    main()