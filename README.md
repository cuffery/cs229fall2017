# cs229fall2017
Stanford CS229 Fall 2017 - Team Project - Real Time Tennis Match Prediction Using Machine Learning

Team: Eddie Chen (yc4@), Yubo Tian (yubotian@), Yi Zhong (yizhon@)

[Project Idea Fair](https://docs.google.com/document/d/1ATAmbQWd25XMRG3Xb3bfEsRBaZRmT6gxRykoD4PvDuA/edit?usp=sharing)

Data generation:
* For player model in current match, set ```G_USE_DIFF``` in .\data_scripts\curr_match_data.py to False, run final_join.py
* For difference model in current match, set ```G_USE_DIFF``` in .\data_scripts\curr_match_data.py to True, run final_join_diff.py

Graph generation:
* After obtaining final_joined_data or final_joined_data_diff, run graph.py

File description:
* logistic.py, svm.py: contains functions related to logistic and svm
* model_selection.py: contains functions related to selecting models with rfe