#   label_set.py
#   
#   Script that reads charting-m-points.csv and generates .csv for results per each set in each match.
#
#   Owner: Yubo Tian
import pandas as pd
import numpy as np
import time
import sys
import math

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
            print("ERROR: label_set: prematurely ended match detected")
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


def getRallyCountSumAfterSet(d, set):
    r = 0
    for i in range(set):
        r += d.iloc[i]
    return r

def prepareRallyCount(match):
    avg_rallyCount = 4 # based on bg research

    # replace NaN with avg_rallyCount, set to unicode str for x.isnumeric() check
    match = match.fillna(str(avg_rallyCount))

    # clean up by dropping non int values, then cast for sum
    match = match[match['rallyCount'].apply(lambda x: x.isnumeric())]

    match['rallyCount'] = match['rallyCount'].astype(int, errors='raise')

    # calc sum of each set, result is in order
    sets = match.groupby(['Set1','Set2'])

    sum_rally_count = sets['rallyCount'].sum()

    #print(sum_rally_count)
    #print(sum_rally_count.iloc[0])

    return sum_rally_count


def processData(data_):
    data = data_[(data_['match_id'].map(lambda x: x.startswith('20')))][['match_id','Set1','Set2','Gm1','Gm2','Pts','Svr','rallyCount']].apply(pd.to_numeric, errors='ignore')

    grouped = data.groupby('match_id')

    r = pd.DataFrame()

    # progress bar 
    sys.stdout.write("=%s]" % (" " * int(len(grouped) / 22)))
    sys.stdout.flush()
    sys.stdout.write("\b" * int(len(grouped) / 22 + 1))
    i = 0
    for match, group in grouped:
        i += 1

        # rally count
        sum_rally_count = prepareRallyCount(group[['match_id','Set1','Set2','rallyCount']])

        num_sets = group.Set1.max() + group.Set2.max() + 1 # +1 for the last set


        for s in range(num_sets):
            # find winner and add an entry for each set in each game
            p = pd.DataFrame({
                'match_id' : grouped.get_group(match).match_id.unique(),
                'set' : s + 1,
                'set_winner' : findWinner(group, s+1),
                'rally_sum' : getRallyCountSumAfterSet(sum_rally_count, s+1)
            })

            r = r.append(p)
        # progress bar
        if ( i % 22 == 0 ):
            sys.stdout.write("=")
            sys.stdout.flush()
    return r


def run(source_dir, out_filename):
    rawData = pd.read_csv(source_dir + 'charting-m-points.csv', delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1", dtype=object)

    start_time = time.time()
    result = processData(rawData)
    #print("--- %s seconds ---" % (time.time() - start_time))

    result.to_csv(out_filename, sep=",",quoting=3, encoding = "ISO-8859-1")

    return

# main()
def main():
    print("[",end="")
    run('../../tennis_MatchChartingProject/', '../result_per_set.csv')
    print("\nDone.")
    return


if __name__ == '__main__':
    main()