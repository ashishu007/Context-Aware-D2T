import argparse
from utils import DataLoader
from tpot import TPOTClassifier
from collections import Counter
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

def main(THEME='streak', DOWN=False):
    FTR = 'num'
    print('Processing data...')

    dl_obj = DataLoader(ftr_type=FTR, downsample=DOWN, theme=THEME)
    train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.get_data()

    print(f"Train: {train_x.shape} {train_y.shape}\tVal: {val_x.shape} {val_y.shape}\tTest: {test_x.shape} {test_y.shape}")
    print(Counter(train_y), Counter(val_y), Counter(test_y))

    print('Finding best classifier with TPOT...')
    tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    tpot.fit(train_x, train_y)
    print(tpot.score(val_x, val_y))
    tpot.export(f'tpot_exports/{THEME}{"_down" if DOWN else ""}.py')
    print(tpot.score(test_x, test_y))
    return "Done!!"

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-down", "--down", help="do downsampling", action="store_true")
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])

    args = argParser.parse_args()
    print(args)

    main(args.theme, args.down)
    print("Done!!!")
