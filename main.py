"""
TODO: add player average clf 
TODO: add team line_score as clf feature - DONE: didnt work
"""

import json 
import pickle
import datasets
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from bert_utils import BertThemeClassifier
from transformers import AutoModelForSequenceClassification
from utils import DataLoader, Classifier, TpotThemeClassifier, RuleClassifier
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

def list_to_dataset(train_x, train_y, val_x, val_y, test_x, test_y):
    train = pd.DataFrame({'text': train_x, 'label': train_y})
    valid = pd.DataFrame({'text': val_x, 'label': val_y})
    test = pd.DataFrame({'text': test_x, 'label': test_y})
    return datasets.DatasetDict({"train": train, "validation": valid, "test": test})

def main(FTR='text', CLF='rf', THEME='streak', DOWNSAMPLE=False, TEST=False, UPSAMPLE=False):
    """
    Main function to run the entire pipeline.
    Parameters:
        FTR: feature type to use. Options --> 'text', 'num'
        CLF: classifier to use. Options --> 'rf', 'if', 'svm', 'bert', 'pet', 'tpot'
        THEME: theme to use. Options --> 'streak', 'standing'
        DOWNSAMPLE: whether to use downsampling on majority class samples or not
        SAVE_TO_DISK: whether to save the data to disk or not
        AUGMENT: whether to apply augmentation (without labels) on minority class and save the data to disk or not
        TEST: whether to just test the classifiers or bothw train and test them
    """

    if CLF == 'if':
        assert DOWNSAMPLE == False, 'Downsampling not supported for Isolation Forest as it is an anomaly detection algorithm'
    elif CLF == 'bert' or CLF == 'pet':
        assert FTR == 'text', f'{CLF.upper()} only works with text features'
    elif CLF == 'if' or CLF == 'svm' or CLF == 'rf' or CLF == 'tpot':
        assert FTR == 'num' or FTR == 'emb', f'{CLF.upper()} only works with numeric or embedded features'
    elif CLF == 'rule':
        assert FTR == 'num', f'{CLF.upper()} only works with numeric features'

    clf_name = f'{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}-up_{"yes" if UPSAMPLE else "no"}'
    dl_obj = DataLoader(downsample=DOWNSAMPLE, theme=THEME, ftr_type=FTR)
    train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.get_data()

    if UPSAMPLE:
        assert FTR == 'num', 'Upsampling only works with numeric features'
        assert DOWNSAMPLE == False, 'Downsampling not supported for upsampling'
        train_x, train_y = dl_obj.upsample_data(train_x, train_y)

    if FTR == 'text':
        print(f"Train: {len(train_x)} {train_y.shape}\tVal: {len(val_x)} {val_y.shape}\tTest: {len(test_x)} {test_y.shape}")
    else:
        print(f"Train: {train_x.shape} {train_y.shape}\tVal: {val_x.shape} {val_y.shape}\tTest: {test_x.shape} {test_y.shape}")
    print(Counter(train_y), Counter(val_y), Counter(test_y))

    if CLF == 'bert' or CLF == 'pet':
        clf_obj = BertThemeClassifier()
        dataset = list_to_dataset(train_x, train_y, val_x, val_y, test_x, test_y)

        if CLF == 'bert':
            print(f'BERT classifier')
            model_path = f"models/{THEME}/{clf_name}"
            if not TEST:
                print('Training BERT model...')
                model, tokenizer = clf_obj.train(dataset)
                model_path = f"models/{THEME}/{clf_name}"
                model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                print(f'Saved model to {model_path}')

        elif CLF == 'pet':
            print('Loading PET model...')
            model_path = f"pet_models/{THEME}/final/p0-i0"

        print('Loading model...')
        trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print('Predicting...')
        test_pred = clf_obj.predict(trained_model, list(test_x))
        print(Counter(test_pred))
    
    elif CLF == 'rule':
        print('Rule classifier')
        clf_obj = RuleClassifier(theme=THEME)
        print('Predicting...')
        test_pred = clf_obj.predict("Model = None", test_x)
        print(Counter(test_pred))

    else:
        if CLF == 'tpot':
            print('This is using tpot model...')
            assert FTR == 'num' or FTR == 'emb', 'TPOT only works with numerical or embedded features'
            clf_obj = TpotThemeClassifier(theme=THEME, downsample=DOWNSAMPLE)
        else:
            clf_obj = Classifier(CLF)

        if not TEST:
            print('Training classifier...')
            model = clf_obj.train(train_x, train_y)
            pickle.dump(model, open(f"models/{THEME}/{clf_name}.pkl", 'wb'))

        print('Predicting...')
        trained_model = pickle.load(open(f"models/{THEME}/{clf_name}.pkl", 'rb'))
        test_pred = clf_obj.predict(trained_model, test_x)

    print(f'Saving test predictions to preds/{THEME}/{clf_name}.npy!!!')
    np.save(open(f'preds/{THEME}/{clf_name}.npy', 'wb'), test_pred)
    print('Calculating performance...')
    test_res_dict, test_clf_report = get_clf_report(test_y, test_pred)

    test_res = {}
    test_res['clf_results'] = test_res_dict
    test_res['class_dist'] = {
            "train": {f"{k}": v for k, v in dict(Counter(train_y)).items()},
            "test": {f"{k}": v for k, v in dict(Counter(test_y)).items()}
        }

    print(f'Test results: {test_res_dict}')
    print(f'Test classification report: {test_clf_report}')

    json.dump(test_res, open(f'results/{THEME}/{clf_name}.json', 'w'), indent=4)
    open(f'results/{THEME}/{clf_name}.txt', 'w').write(test_clf_report)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-up", "--up", help="do upsampling", action="store_true")
    argParser.add_argument("-down", "--down", help="do downsampling", action="store_true")
    argParser.add_argument("-test", "--test", help="do only testing and not trainign", action="store_true")
    argParser.add_argument("-ftr", "--ftr", help="feature type: numerical or textual", default="text", choices=["text", "num", "emb"])
    argParser.add_argument("-clf", "--clf", help="classification algorithm", default="rf", \
                            choices=["rf", "svm", "bert", "if", "pet", 'tpot', 'knn', 'ann', 'lr', 'rule'])
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])

    args = argParser.parse_args()
    print(args)

    main(args.ftr, args.clf, args.theme, args.down, args.test, args.up)
    print("Done!!!")

