import pandas as pd
from utils import DataLoader
from collections import Counter
from team_themes import TeamStreak, TeamStanding

FTR = 'num'

for THEME in ['standing']:#, 'streak']:

    if THEME == 'streak':
        theme_obj = TeamStreak(FTR)
    elif THEME == 'standing':
        theme_obj = TeamStanding(FTR)

    print('Loading all data...')
    all_data, aug_texts = theme_obj.get_theme_train_val_test_data()

    for DOWN in [True, False]:
        print(f'{THEME} {DOWN}')

        dl_obj = DataLoader(ftr_type=FTR, theme=THEME, downsample=DOWN)
        train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.prep_data(all_data)
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
