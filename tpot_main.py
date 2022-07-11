import argparse
from numpy import std
import pandas as pd
from utils import DataLoader
from tpot import TPOTClassifier
from collections import Counter
from team_themes import TeamStanding, TeamStreak
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, classification_report


def get_clf_report(true, pred):
    mf1 = float(f"{f1_score(true, pred, average='macro'):.2f}")
    acc = float(f"{accuracy_score(true, pred):.2f}")
    f1 = float(f"{f1_score(true, pred):.2f}")
    prec = float(f"{precision_score(true, pred):.2f}")
    rec = float(f"{recall_score(true, pred):.2f}")
    fbeta = float(f"{fbeta_score(true, pred, beta=2):.2f}")
    return {
        'mf1': mf1,
        'acc': acc,
        'f1': f1,
        'prec': prec,
        'rec': rec,
        'fbeta': fbeta
    }, classification_report(true, pred)


def save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN=False):
    print('Saving data to disk into csv...')
    num_ftrs = train_x.shape[1]
    train = pd.DataFrame({"text": train_x, "label": list(train_y)})
    val = pd.DataFrame({"text": val_x, "label": list(val_y)})
    test = pd.DataFrame({"text": test_x, "label": list(test_y)})
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    train.to_csv(f'data/{THEME}/train_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    val.to_csv(f'data/{THEME}/validation_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    test.to_csv(f'data/{THEME}/test_text{"_down" if DOWN else ""}.csv', index=False, header=False)


def main(THEME='streak', DOWN=False, STD=False):
    FTR = 'num'
    if THEME == 'streak':
        theme_obj = TeamStreak(FTR)
    elif THEME == 'standing':
        theme_obj = TeamStanding(FTR)

    print('Processing data...')
    all_data, aug_texts = theme_obj.get_theme_train_val_test_data()
    dl_obj = DataLoader(ftr_type=FTR, downsample=DOWN)
    train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.prep_data(all_data)
    print(f"Train: {train_x.shape} {train_y.shape}\tVal: {val_x.shape} {val_y.shape}\tTest: {test_x.shape} {test_y.shape}")
    print(Counter(train_y), Counter(val_y), Counter(test_y))

    if STD:
        # save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN)
        return

    print('Finding best classifier with TPOT...')
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(train_x, train_y)
    print(tpot.score(val_x, val_y))
    tpot.export(f'tpot_exports/{THEME}{"_down" if DOWN else ""}.py')
    print(tpot.score(test_x, test_y))

    return "Done!!"


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-do_down", "--do_down", help="do downsampling", action="store_true")
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])
    argParser.add_argument("-std", "--std", help="save data to disk", action="store_true")

    args = argParser.parse_args()
    print(args)

    main(args.theme, args.do_down, args.std)
    print("Done!!!")
