#   label_set.py
#   
#   Script that reads charting-m-points.csv and generates .csv for results per each set in each match.
#
#   Owner: Yubo Tian
import pandas as pd
import numpy as np
import time

# main()
def main():
    rawData = readFile('../tennis_MatchChartingProject/charting-m-points.csv')

    start_time = time.time()

    print("--- %s seconds ---" % (time.time() - start_time))

    return

if __name__ == '__main__':
    main()