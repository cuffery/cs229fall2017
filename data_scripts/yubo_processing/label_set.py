#   label_set.py
#   
#   Script that reads charting-m-points.csv and generates .csv for results per each set in each match.
#
#   Owner: Yubo Tian
import pandas as pd
import numpy as np
import time


def findFinalSetWinner(group):
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
                print ('unexpected match result')
    return winner


def findWinner(matchData, currSet):
    set_scores = matchData[['Set1','Set2']].drop_duplicates()
    if (currSet == set_scores.shape[0]):
        return findFinalSetWinner(matchData)
    else:
        # not the final set, compare this set's score with the prev one (i.e. 2-1 compared with 2-0 or 1-1)
        winner = 0

        # catch the matches that ends before first set ends (i.e. only set_score = 0-0)
        if (set_scores.shape[0] == 1):
            print("prematurely ended match detected")
            return 1

        # find the current set (and the prev set)
        curr = set_scores.iloc[currSet]
        prev = set_scores.iloc[currSet-1]

        if ((curr.Set1-curr.Set2) - (prev.Set1-prev.Set2) == 1):
            winner = 1
        elif ((curr.Set1-curr.Set2) - (prev.Set1-prev.Set2) == -1):
            winner = 2
        else:
            print ("unexpected set result")

        return winner


def processData(data_):
    data = data_[(data_['match_id'].map(lambda x: x.startswith('20')))][['match_id','Set1','Set2','Gm1','Gm2','Pts','Svr']].apply(pd.to_numeric, errors='ignore')

    grouped = data.groupby('match_id')

    r = pd.DataFrame()
    for match, group in grouped:
        num_sets = group.Set1.max() + group.Set2.max() + 1 # +1 for the last set
        for s in range(num_sets):
            # find winner and add an entry for each set in each game
            p = pd.DataFrame({
                'match_id' : grouped.get_group(match).match_id.unique(),
                'set' : s + 1,
                'set_winner' : findWinner(group, s+1)
            })

            r = r.append(p)
    return r


# main()
def main():
    rawData = pd.read_csv('../tennis_MatchChartingProject/charting-m-points.csv', delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1", dtype=object)

    start_time = time.time()
    result = processData(rawData)
    print("--- %s seconds ---" % (time.time() - start_time))

    result.to_csv('result_per_set.csv', sep=",",quoting=3, encoding = "ISO-8859-1")

    return


if __name__ == '__main__':
    main()