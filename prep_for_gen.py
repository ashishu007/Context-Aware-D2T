import json
import pickle
import argparse
from tqdm import tqdm 
from datasets import load_dataset
from team_themes import TeamStanding, TeamStreak
from player_themes import PlayerAverage, PlayerDouble

argparser = argparse.ArgumentParser()
argparser.add_argument('--part', '-part', type=str, default='test', choices=['test', 'train', 'validation'])
args = argparser.parse_args()
part = args.part

print(f'Arguments: {args}')
print(f'Part: {part}')

dataset = load_dataset('GEM/sportsett_basketball')
phkeys_id = {'PTS': 1, 'AST': 2, 'TREB': 3, 'BLK': 4, 'STL': 5}
phkeys_id_rev = {1: 'PTS', 2: 'AST', 3: 'TREB', 4: 'BLK', 5: 'STL'}

streak_obj = TeamStreak(ftr_type='num')
double_obj = PlayerDouble(ftr_type='num')
average_obj = PlayerAverage(ftr_type='num')
standing_obj = TeamStanding(ftr_type='num')

double_clf = pickle.load(open(f'models/double/rf-num-down_yes-up_no.pkl', 'rb'))
streak_clf = pickle.load(open(f'models/streak/rf-num-down_yes-up_no.pkl', 'rb'))
average_clf = pickle.load(open(f'models/average/rf-num-down_yes-up_no.pkl', 'rb'))
standing_clf = pickle.load(open(f'models/standing/rf-num-down_yes-up_no.pkl', 'rb'))

double_hist = json.load(open(f'doubles/{part}.json', 'r'))
streak_hist = json.load(open(f'streaks/{part}.json', 'r'))
standing_hist = json.load(open(f'standings/{part}.json', 'r'))
average_hist = json.load(open(f'average_stats/{part}.json', 'r'))
broken_streak_hist = json.load(open(f'streaks/broken_streak-{part}.json', 'r'))

all_hists = {}
for idx, item in enumerate(tqdm(dataset[f'{part}'])):
    # if idx == 2:
    #     break

    item_hists = {
        "home": {
            "ls": {"streak": "", "standing": ""}, 
            "bs": {"average": {}, "double": {}}
        }, 
        "vis": {
            "ls": {"streak": "", "standing": ""},
            "bs": {"average": {}, "double": {}}
        }
    }

    item_gemid = item['gem_id']
    hname, hplace = item['teams']['home']['name'], item['teams']['home']['place']
    vname, vplace = item['teams']['vis']['name'], item['teams']['vis']['place']
    hteam = f"{hplace} {hname}"
    vteam = f"{vplace} {vname}"

    hls, vls = item['teams']['home']['line_score']['game'], item['teams']['vis']['line_score']['game']
    hbs, vbs = item['teams']['home']['box_score'], item['teams']['vis']['box_score']

    # Team Streak Info
    hstreak_info = streak_hist[item_gemid]['home']
    hbroken_streak_info = {}
    if item_gemid in broken_streak_hist:
        if 'home' in broken_streak_hist[item_gemid]:
            hbroken_streak_info = broken_streak_hist[item_gemid]['home']
    vstreak_info = streak_hist[item_gemid]['vis']
    vbroken_streak_info = {}
    if item_gemid in broken_streak_hist:
        if 'vis' in broken_streak_hist[item_gemid]:
            vbroken_streak_info = broken_streak_hist[item_gemid]['vis']

    hstreak_ftrs = streak_obj.team_streak_num_ftr(hstreak_info, hbroken_streak_info, hteam, hls)
    vstreak_ftrs = streak_obj.team_streak_num_ftr(vstreak_info, vbroken_streak_info, vteam, vls)

    hstreak_pred = streak_clf.predict([hstreak_ftrs])[0]
    vstreak_pred = streak_clf.predict([vstreak_ftrs])[0]

    hstreak_str = f"<{'WINNING' if hstreak_ftrs[1] == 1 else 'LOSING'}-STREAK> {hstreak_ftrs[2]}"
    vstreak_str = f"<{'WINNING' if vstreak_ftrs[1] == 1 else 'LOSING'}-STREAK> {vstreak_ftrs[2]}"

    # if part == 'test':
    item_hists['home']['ls']['streak'] = hstreak_str if hstreak_pred == 1 else "None"
    item_hists['vis']['ls']['streak'] = vstreak_str if vstreak_pred == 1 else "None"

    # Team Standing Info
    hstanding_info = standing_hist[item_gemid]['home']
    vstanding_info = standing_hist[item_gemid]['vis']

    hstanding_ftrs = standing_obj.team_standing_num_ftr(hstanding_info, hls)
    vstanding_ftrs = standing_obj.team_standing_num_ftr(vstanding_info, vls)

    hstanding_pred = standing_clf.predict([hstanding_ftrs])[0]
    vstanding_pred = standing_clf.predict([vstanding_ftrs])[0]

    hstanding_str = f"<CONF-STANDING> {hstanding_ftrs[1]}"
    vstanding_str = f"<CONF-STANDING> {vstanding_ftrs[1]}"

    # if part == 'test':
    item_hists['home']['ls']['standing'] = hstanding_str if hstanding_pred == 1 else "None"
    item_hists['vis']['ls']['standing'] = vstanding_str if vstanding_pred == 1 else "None"

    # Player Double Info
    for player in hbs:
        player_name = player['name']
        player_side = 'home'
        player_avg_info = average_hist[item_gemid][player_side][player_name]
        player_double_info = double_hist[item_gemid][player_side][player_name]
        player_avg_ftrs = average_obj.player_average_num_ftr(player_name, player_avg_info)
        player_double_ftrs = double_obj.player_double_num_ftr(player_name, player_double_info)
        player_avg_pred = average_clf.predict(player_avg_ftrs) # because this will have many rows
        player_double_pred = double_clf.predict([player_double_ftrs])[0]

        # ftrs = [player_id, straight_dd, straight_td, total_dd, total_td, double_in_last_5, triple_in_last_5]
        double_str = f"<STR-DD/TD> {player_double_ftrs[1]} {player_double_ftrs[2]} <TOTAL-DD/TD> {player_double_ftrs[3]} {player_double_ftrs[4]} <LAST_5-DD/TD> {player_double_ftrs[5]} {player_double_ftrs[6]}"
        if player_double_ftrs[1] == 0 and player_double_ftrs[2] == 0 and player_double_ftrs[3] == 0 and player_double_ftrs[4] == 0 and player_double_ftrs[5] == 0 and player_double_ftrs[6] == 0:
            double_str = "None" 
        else:
            double_str = double_str
        # if part == 'test':
        item_hists['home']['bs']['double'][player_name] = double_str if player_double_pred == 1 else "None"

        player_avg_str = ""
        for avg_ftr, avg_pred in zip(player_avg_ftrs, player_avg_pred):
            # ftrs: [player_id, pts/rebounds/assists/steals/blocks, last_Y_games, value, avg/total]
            last_y_str = f"LAST-{avg_ftr[2]}-GAMES" if avg_ftr[2] != 100 else "SEASON"
            if avg_ftr[3] < 5:
                player_avg_str = f"{player_avg_str}"
            else:
                player_avg_str = f"{player_avg_str} <{'AVG' if avg_ftr[-1] == 1 else 'TOTAL'}-{phkeys_id_rev[avg_ftr[1]]}-{last_y_str}> {avg_ftr[3]}"
        # if part == 'test':
        item_hists['home']['bs']['average'][player_name] = player_avg_str.strip() if avg_pred == 1 else "None"

    for player in vbs:
        player_name = player['name']
        player_side = 'vis'
        player_avg_info = average_hist[item_gemid][player_side][player_name]
        player_double_info = double_hist[item_gemid][player_side][player_name]
        player_avg_ftrs = average_obj.player_average_num_ftr(player_name, player_avg_info)
        player_double_ftrs = double_obj.player_double_num_ftr(player_name, player_double_info)
        player_avg_pred = average_clf.predict(player_avg_ftrs) # because this will have many rows
        player_double_pred = double_clf.predict([player_double_ftrs])[0]

        # ftrs = [player_id, straight_dd, straight_td, total_dd, total_td, double_in_last_5, triple_in_last_5]
        double_str = f"<STR-DD/TD> {player_double_ftrs[1]} {player_double_ftrs[2]} <TOTAL-DD/TD> {player_double_ftrs[3]} {player_double_ftrs[4]} <LAST_5-DD/TD> {player_double_ftrs[5]} {player_double_ftrs[6]}"
        if player_double_ftrs[1] == 0 and player_double_ftrs[2] == 0 and player_double_ftrs[3] == 0 and player_double_ftrs[4] == 0 and player_double_ftrs[5] == 0 and player_double_ftrs[6] == 0:
            double_str = "None" 
        else:
            double_str = double_str
        # if part == 'test':
        item_hists['vis']['bs']['double'][player_name] = double_str if player_double_pred == 1 else "None"

        player_avg_str = ""
        for avg_ftr, avg_pred in zip(player_avg_ftrs, player_avg_pred):
            # ftrs: [player_id, pts/rebounds/assists/steals/blocks, last_Y_games, value, avg/total]
            last_y_str = f"LAST-{avg_ftr[2]}-GAMES" if avg_ftr[2] != 100 else "SEASON"
            if avg_ftr[3] < 5:
                player_avg_str = f"{player_avg_str}"
            else:
                player_avg_str = f"{player_avg_str} <{'AVG' if avg_ftr[-1] == 1 else 'TOTAL'}-{phkeys_id_rev[avg_ftr[1]]}-{last_y_str}> {avg_ftr[3]}"
        # if part == 'test':
        item_hists['vis']['bs']['average'][player_name] = player_avg_str.strip() if avg_pred == 1 else "None"

    all_hists[item_gemid] = item_hists

json.dump(all_hists, open(f'preds/{part}_hist_preds.json', 'w'), indent='\t')

