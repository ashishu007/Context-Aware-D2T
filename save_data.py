import argparse
import pandas as pd
from utils import PrepData
from collections import Counter
from team_themes import TeamStreak, TeamStanding
from player_themes import PlayerAverage, PlayerDouble

argparser = argparse.ArgumentParser()
argparser.add_argument('--ftr', '-ftr', type=str, default='num', help='Feature type to use.', choices=['text', 'num', 'emb'])
argparser.add_argument('--theme', '-theme', type=str, default='streak', help='Theme to use.', choices=['streak', 'standing', 'average', 'double'])
argparser.add_argument('--down', '-down', action='store_true', help='Whether to use downsampling on majority class samples or not')

args = argparser.parse_args()
print(args)

FTR = args.ftr
THEME = args.theme
DOWNSAMPLE = args.down
print(f'FTR: {FTR}\tTHEME: {THEME}\tDOWNSAMPLE: {DOWNSAMPLE}')

if THEME == 'streak':
    theme_obj = TeamStreak(FTR)
elif THEME == 'standing':
    theme_obj = TeamStanding(FTR)
elif THEME == 'average':
    theme_obj = PlayerAverage(FTR)
elif THEME == 'double':
    theme_obj = PlayerDouble(FTR)

print('Loading all data...')
all_data, aug_texts = theme_obj.get_theme_train_val_test_data()

for DOWN in [True, False]:
    print(f'{THEME} {DOWN}')

    pd_obj = PrepData(ftr_type=FTR, theme=THEME, downsample=DOWN)
    train_x, train_y, val_x, val_y, test_x, test_y = pd_obj.prep_data(all_data)
    print(f"Train: {train_x.shape} {train_y.shape}\tVal: {val_x.shape} {val_y.shape}\tTest: {test_x.shape} {test_y.shape}")
    print(Counter(train_y), Counter(val_y), Counter(test_y))

    print(f'Saving data for {FTR} features to disk into csv...')

    if FTR == 'text':
        train = pd.DataFrame({"text": train_x, "label": list(train_y)})
        val = pd.DataFrame({"text": val_x, "label": list(val_y)})
        test = pd.DataFrame({"text": test_x, "label": list(test_y)})
    elif FTR == 'num':
        train = pd.DataFrame(train_x)
        train['y'] = train_y
        val = pd.DataFrame(val_x)
        val['y'] = val_y
        test = pd.DataFrame(test_x)
        test['y'] = test_y

    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    train.to_csv(f'data/{THEME}/train_{FTR}{"_down" if DOWN else ""}.csv', index=False, header=False)
    val.to_csv(f'data/{THEME}/validation_{FTR}{"_down" if DOWN else ""}.csv', index=False, header=False)
    test.to_csv(f'data/{THEME}/test_{FTR}{"_down" if DOWN else ""}.csv', index=False, header=False)
