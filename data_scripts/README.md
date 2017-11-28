# cs229fall2017/data_scripts
Each one of our sample data (i.e. each x<sup>(i)</sup>) can be viewed as containing three components:

* [match_info] historical data - general info about the match (tournament info, match surface, weather, etc)

* [past_perf] historical data - data on players's past performance, calsulated using the common opponent model

* [in_game_data] real-time data - data on players' detailed performance during the match of interest, i.e. the result of which we are trying to predict with our model 

# Acknowledgment
Each sub folder in data_scripts dir stores the scripts we wrote to clean and process the corresponding components described above:

* ./match_info/ owner: Yi Zhong (yizhon@), data source: [tennis_MatchChartingProject by JeffSackmann](https://github.com/JeffSackmann/tennis_MatchChartingProject)

* ./past_perf/ owner: Eddie Chen (yc4@), data source: [tennis_atp by JeffSackmann](https://github.com/JeffSackmann/tennis_atp/tree/ef3eaf0320e60bcedc9d00933c00d4a5893b972e)

* ./in_game_data/ owner: Yubo Tian (yubotian@), data source: [tennis_MatchChartingProject by JeffSackmann](https://github.com/JeffSackmann/tennis_MatchChartingProject)

Scripts that were used to join the processed three sections of data into a complete data set live in this shared directory /data_scripts.
