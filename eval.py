import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.metrics import precision_score, recall_score, classification_report

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


clfs = ['rf', 'if', 'svm', 'bert', 'pet']
ftrs = ['text', 'num']
downs = ['yes', 'no']
for theme in ['streak', 'standing']:
    for clf in clfs:
        for ftr in ftrs:
            for do_down in downs:
                print(f'Theme: {theme}\tClassifier: {clf}\tFeature: {ftr}\tDownsampling: {do_down}')
                if clf == 'bert' and ftr == 'num':
                    continue
                elif clf == 'if' and do_down == 'yes':
                    continue
                elif (clf == 'pet' and ftr == 'num') or (clf == 'pet' and do_down == 'yes'):
                    continue
                else:
                    true = np.array(pd.read_csv(f'data/{theme}/test_text.csv', names=['text', 'label'])['label'].to_list())
                    pred = np.load(f'preds/{theme}/{clf}-{ftr}-down_{do_down}.npy')
                    print(true.shape, pred.shape)
                    # res_dict, report_str = get_clf_report(true, pred)
                    # print(res_dict)

