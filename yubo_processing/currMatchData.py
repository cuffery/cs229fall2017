#   CurrentMatchData.py
#   
#   Script that read match data and generates .csv for current match status.
#
#   Owner: Yubo Tian
import pandas as pd

# aggregateOnSet(data)
def aggregateOnSet(data):
    df = pd.DataFrame()
    # e.g. df['color'] = np.where(((df.A < borderE) & ((df.B - df.C) < ex)), 'r', 'b')

    return df


# readFile(filename)
def readFile(filename):
    rawData = pd.read_csv(filename, delimiter=",", quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
    return rawData;


# debug(data)
# description: container fucntion for testing/ sample code
def debug(data):
    print(data.iloc[0])
    # total num of matches
    # print(rawData.match_id.unique().size)
    
    # num of matches in or after year 2000
    # criteria = rawData['match_id'].map(lambda x: x.startswith('20'))
    # matches2017 = rawData[criteria]
    # print(matches2017.match_id.unique().size)

    return


# main()
def main():
    rawData = readFile('../tennis_MatchChartingProject/charting-m-stats-Overview.csv')
    debug(rawData)

    aggregateOnSet(rawData)

    return

if __name__ == '__main__':
    main()