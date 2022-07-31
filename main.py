"""
TODO: add team popularity score
"""

import json 
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter
from utils import DataLoader, Classifier, Sampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import fbeta_score, precision_score
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.metrics import recall_score, classification_report

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
    }, classification_report(true, pred), confusion_matrix(true, pred)

def plot_cm(cm, theme='standing', season='all', part='test'):
    df = pd.DataFrame(cm, index=['yes', 'no'], columns=['yes', 'no'])
    plt.figure(figsize=(10,7))
    plt.rcParams['font.size'] = 18
    sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{theme.upper()} Confusion Matrix")
    tick_marks = [0.5, 1.5] 
    plt.xticks(tick_marks, ['No', 'Yes'], rotation=0)
    plt.yticks(tick_marks, ['No', 'Yes']) 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f"results/{part}/{theme}-{season}.png", dpi=300)

def get_opt_threshold(test_y, pred_probas):
    fpr, tpr, thresholds = roc_curve(test_y, pred_probas[:, 1])
    gmean = np.sqrt(tpr * (1 - fpr))
    roc_auc = auc(fpr, tpr)
    index = np.argmax(gmean)
    thresholdOptg = round(thresholds[index], ndigits = 4)
    youdenJ = tpr - fpr
    index = np.argmax(youdenJ)
    thresholdOpty = round(thresholds[index], ndigits = 4)
    return thresholdOptg, thresholdOpty, fpr, tpr, roc_auc

def main(THEME='streak', SEASON='all'):
    """
    Main function to run the entire pipeline.
    Parameters:
        THEME: theme to use. Options --> 'streak', 'standing'
    """

    dl_obj = DataLoader(theme=THEME, season=SEASON)
    train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.get_data()

    clfs = ['rf', 'knn', 'lr']
    samplers = ['none', 'smote', 'adasyn', 'svmsmote', 'smoten', 'borderlinesmote', 'kmeanssmote' \
                'randomoversampler', 'randomundersampler', 'allknn', 'repeatededitednearestneighbours', \
                'instancehardnessthreshold', 'nearmiss', 'onesidedselection', 'tomeklinks', 'editednearestneighbours']

    best_val, best_comb = 0, {'clf': 'rf', 'sampler': 'none', 'threshold': 0}
    for clf in clfs:
        clf_obj = Classifier(algo_name=clf)
        print(f"This is the {clf.upper()} classifier for {THEME.upper()} theme!!!")
        for sampler in tqdm(samplers):
            sampler_obj = Sampler(method=sampler)
            train_x, train_y, msg = sampler_obj.sample_data(train_x, train_y)
            if msg == 'failure':
                continue
            model = clf_obj.train(train_x, train_y)
            pred_probas = clf_obj.predict_proba(model, val_x)
            # print(pred_probas.shape, sampler, clf, train_y.shape, Counter(train_y))
            thresholdOptg, thresholdOpty, _, _, _ = get_opt_threshold(val_y, pred_probas)
            for thresh_val in [thresholdOptg, thresholdOpty]:
                pred_y = np.where(pred_probas[:, 1] > thresh_val, 1, 0)
                res_dict, _, _ = get_clf_report(val_y, pred_y)
                if res_dict['mf1'] > best_val:
                    best_val = res_dict['mf1']
                    best_comb = {'clf': clf, 'sampler': sampler, 'threshold': thresh_val}
    print(f"Best combination is {best_comb} with {best_val} MacroF1 score on validation set!!!")

    sampler_obj = Sampler(method=best_comb['sampler'])
    train_x, train_y, _ = sampler_obj.sample_data(train_x, train_y)

    mf1s = {'test': [], 'val': []}
    for random_seed in [0, 123, 456, 789, 1024, 4201, 987, 654, 321, 10]:
        clf_obj = Classifier(algo_name=best_comb['clf'], random_seed=random_seed)
        model = clf_obj.train(train_x, train_y)

        pred_probas_test = clf_obj.predict_proba(model, test_x)
        pred_y_test = np.where(pred_probas_test[:, 1] > best_comb['threshold'], 1, 0)
        res_dict_test, _, _ = get_clf_report(test_y, pred_y_test)
        mf1s['test'].append(res_dict_test['mf1'])

        pred_probas_val = clf_obj.predict_proba(model, val_x)
        pred_y_val = np.where(pred_probas_val[:, 1] > best_comb['threshold'], 1, 0)
        res_dict_val, _, _ = get_clf_report(val_y, pred_y_val)
        mf1s['val'].append(res_dict_val['mf1'])

    for part, pred_y in {"test": pred_y_test, "val": pred_y_val}.items():
        true_y = test_y if part == 'test' else val_y
        res_dict, clf_report, conf_matrix = get_clf_report(true_y, pred_y)
        plot_cm(conf_matrix, theme=THEME, season=SEASON, part=part)
        open(f'results/{part}/{THEME}-{SEASON}.txt', 'w').write(clf_report)
        test_res = {}
        test_res['clf_results'] = res_dict
        test_res['best_comb'] = best_comb
        test_res['mean_mf1'] = float(np.mean(mf1s[f'{part}']))
        test_res['std_mf1'] = float(np.std(mf1s[f'{part}']))
        test_res['mf1s'] = mf1s[f'{part}']
        test_res['class_dist'] = {
            "train": {f"{k}": v for k, v in dict(Counter(train_y)).items()},
            "test": {f"{k}": v for k, v in dict(Counter(true_y)).items()}
        }
        json.dump(test_res, open(f'results/{part}/{THEME}-{SEASON}.json', 'w'), indent=4)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])
    argParser.add_argument("-season", "--season", help="season to use for train/valid/test set data", default='all', \
                            choices=['all', 'bens', 'juans', 'train_set', '2014', '2015', '2016', '2017', '2018', \
                                    'carlos', 'dans', 'joels', 'oscars'])

    args = argParser.parse_args()
    print(args)

    main(args.theme, args.season)
    print("Done!!!")

