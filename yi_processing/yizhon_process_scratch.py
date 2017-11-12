import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import scipy
import pandas as pd

#match_data = pd.read_csv('charting-m-matches.csv')
match_data = pd.read_csv('charting-m-matches.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")
point_data = pd.read_csv('charting-m-points.csv', delimiter=",",quoting=3, error_bad_lines=False, encoding = "ISO-8859-1")

#train = np.genfromtxt('charting-m-matches.csv', dtype = mystr , skip_header=True, delimiter=',')

print(match_data.iloc[1])
print(point_data.iloc[1])

#print(train)

#match_data = genfromtxt('charting-m-matches.csv', delimiter=',')
#print(match_data)