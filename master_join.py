import numpy as np
import pandas as pd

joined_hist_match = pd.read_csv('../ec_processing/joined_hist_match.csv',
                    index_col=None,
                    header=0)

atp_label = pd.read_csv('../ec_processing/match_label_ATP.csv',
                        index_col=None,
                        header=0)

curr_match_data = pd.read_csv('../yubo_processing/current_match_data.csv',
                        index_col=None,
                        header=0)

print(list(joined_hist_match))
print(list(atp_label))
print(list(curr_match_data))