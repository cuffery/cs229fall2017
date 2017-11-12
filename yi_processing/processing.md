Training data:
- A table with unique match_id + set number number rows, with corresponding measurements computed from original data

stats - charting-m-stats-Overview
netpoints - charting-m-stats-NetPoints
matches - charting-m-matches
players - apt_players
points - charting-m-points

Table 2.1: OnCourt dataset
Player details
    - players - (column 3) Dominant hand
Name
    - players or matches both have this info
players - (column 4) Date of birth
players - (column 5) Country of birth
Prize money
ATP rating points over time
    - atp_rankings_10s (column 4), need to match with atp_players id
ATP rank over time
    - atp_rankings_10s (column 1), need to match with atp_players id

Match details
matches - (Tournament) Tournament name
Tournament type (e.g., Grand Slam)
matches - (Surface) Surface
Location (country, lat/lon)
matches - (Date) Date
Result (scoreline)
Prize money
Odds (Marathonbet, Pinnacle)

Per-set stats for both players
stats - (first_in/serve_pts)First serve percentage
stats - (aces) Aces
stats - (dfs) Double faults
stats - (unforced) Unforced errors
stats - (first_won/first_in) Percentage of points won on first serve
stats - (second_won/second_in)Percentage of points won on second serve
stats - (return_pts_won/return_pts)Percentage of receiving points won
stats - (winners) Winners
stats - (bk_pts) Break points (won, total)
netpoints - (pts_won/net_pts)Net approaches (won, total)
stats - (second_won + first_won + return_pts_won/serve_pts + return_pts)Total points won
Fastest serve
Average first serve speed
Average second serve speed
Odds (Marathonbet, Pinnacle)

New features:
- Round
- L/R hand