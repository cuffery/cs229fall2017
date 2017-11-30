#   label_match.py
#   
#   Script that reads charting-m-points.csv and generates .csv for results per each match.
#
#   Owner: Yubo Tian
import pandas as pd
import numpy as np
import time

def processData(data_):
    # rm matches before 2000 and keep only relevant cols
    data = data_[(data_['match_id'].map(lambda x: x.startswith('20')))][['match_id','Set1','Set2','Gm1','Gm2','Pts','Svr']].apply(pd.to_numeric, errors='ignore')

    grouped = data.groupby('match_id')

    r = pd.DataFrame()

    for match, group in grouped:
        winner = 0
        if group.Set1.max() > group.Set2.max():
            winner = 1
        elif group.Set1.max() < group.Set2.max():
            winner = 2
        else:
            if (group.Gm1.iloc[-1] > group.Gm2.iloc[-1]):
                 winner = 1
            elif (group.Gm1.iloc[-1] < group.Gm2.iloc[-1]):
                 winner = 2
            else:
                server_pts = int(group.Pts.iloc[-1].split('-')[0])
                receiver_pts = int(group.Pts.iloc[-1].split('-')[-1])
                if (server_pts > receiver_pts):
                    winner = group.Svr.iloc[-1]
                elif (server_pts < receiver_pts):
                    winner = group.Svr.iloc[-1] * -1 + 3
                else:
                    print ('ERROR: unexpected match result')

        p = pd.DataFrame({
            'match_id' : grouped.get_group(match).match_id.unique(),
            'match_winner' : winner
        })

        r = r.append(p)
    return r

def run(source_dir, out_filename):
    rawData = pd.read_csv(source_dir + 'charting-m-points.csv', delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1", dtype=object)

    print('[', end='') # hack for progress bar when running final_join.py, dont remove

    start_time = time.time()
    result = processData(rawData)
    #print("--- %s seconds ---" % (time.time() - start_time))

    result.to_csv(out_filename, sep=",",quoting=3, encoding = "ISO-8859-1")

    return

# main()
def main():
    run('../../tennis_MatchChartingProject/','../result_per_match.csv')
    return

if __name__ == '__main__':
    main()