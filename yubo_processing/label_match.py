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
    data = data_[(data_['match_id'].map(lambda x: x.startswith('20')))][['match_id','Set1','Set2']].apply(pd.to_numeric, errors='ignore')

    grouped = data.groupby('match_id')

    r = pd.DataFrame()

    for match, group in grouped:
        p = pd.DataFrame({
            'match_id' : grouped.get_group(match).match_id.unique(),
            'match_winner' : 1 if group.Set1.max() > group.Set2.max() else 2
         })

        r = r.append(p)
    return r


# main()
def main():
    rawData = pd.read_csv('../tennis_MatchChartingProject/charting-m-points.csv', delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1", dtype=object)

    start_time = time.time()
    result = processData(rawData)
    print("--- %s seconds ---" % (time.time() - start_time))

    result.to_csv('result_per_match.csv', sep=",",quoting=3, encoding = "ISO-8859-1")

    return

if __name__ == '__main__':
    main()