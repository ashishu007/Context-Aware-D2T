import numpy as np
import pandas as pd
import argparse, json, pickle
from collections import Counter
from team_themes import TeamStanding, TeamStreak
from utils import DataLoader, Classifier, CalcPerformance
from bert_utils import BertDataLoader, BertThemeClassifier
from transformers import AutoModelForSequenceClassification

def save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN=False):
    print('Saving data to disk into csv...')
    train = pd.DataFrame({"text": train_x, "label": list(train_y)})
    val = pd.DataFrame({"text": val_x, "label": list(val_y)})
    test = pd.DataFrame({"text": test_x, "label": list(test_y)})
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    train.to_csv(f'data/{THEME}/train_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    val.to_csv(f'data/{THEME}/validation_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    test.to_csv(f'data/{THEME}/test_text{"_down" if DOWN else ""}.csv', index=False, header=False)


def main(FTR='text', CLF='rf', THEME='streak', DOWNSAMPLE=False, SAVE_TO_DISK=False):

    if CLF == 'if':
        assert DOWNSAMPLE == False, 'Downsampling not supported for Isolation Forest as it is an anomaly detection algorithm'
    elif CLF == 'bert':
        assert FTR == 'text', 'Bert only works with text features'

    if THEME == 'streak':
        theme_obj = TeamStreak(FTR)
    elif THEME == 'standing':
        theme_obj = TeamStanding(FTR)

    print('Processing data...')
    all_data = theme_obj.get_theme_train_val_test_data()

    if CLF == 'bert':
        clf_obj = BertThemeClassifier()
        dl_obj = BertDataLoader(downsample=DOWNSAMPLE)

        print('Preprocessing data...')
        dataset, train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.prep_data(all_data)
        print(f"Train: {len(train_x)} {train_y.shape}\tVal: {len(val_x)} {val_y.shape}\tTest: {len(test_x)} {test_y.shape}")
        print(Counter(train_y), Counter(val_y), Counter(test_y))

        if SAVE_TO_DISK:
            save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN=DOWNSAMPLE)
            return 

        trained_model, tokenizer = clf_obj.train(dataset)

        model_save_path = f"models/{THEME}/{CLF}-down_{'yes' if DOWNSAMPLE else 'no'}"
        trained_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        print('Loading model...')
        trained_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
        print('Predicting...')
        test_pred = clf_obj.predict(trained_model, list(test_x))
        print(Counter(test_pred))

    else:
        clf_obj = Classifier(CLF)
        dl_obj = DataLoader(ftr_type=FTR, downsample=DOWNSAMPLE)

        train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.prep_data(all_data)
        print(f"Train: {train_x.shape} {train_y.shape}\tVal: {val_x.shape} {val_y.shape}\tTest: {test_x.shape} {test_y.shape}")
        print(Counter(train_y), Counter(val_y), Counter(test_y))

        if SAVE_TO_DISK:
            save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN=DOWNSAMPLE)
            return 

        print('Training classifier...')
        model = clf_obj.train(train_x, train_y)
        pickle.dump(model, open(f'models/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.pkl', 'wb'))

        print('Predicting...')
        # val_pred = clf_obj.predict(model, val_x)
        test_pred = clf_obj.predict(model, test_x)

    np.save(open(f'preds/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.npy', 'wb'), test_pred)
    print('Calculating performance...')
    per_obj = CalcPerformance()
    # val_res_dict, val_clf_report = clf_obj.get_clf_report(val_y, val_pred)
    test_res_dict, test_clf_report = per_obj.get_clf_report(test_y, test_pred)

    test_res = {}
    test_res['clf_results'] = test_res_dict
    test_res['class_dist'] = {
            "train": {f"{k}": v for k, v in dict(Counter(train_y)).items()},
            "test": {f"{k}": v for k, v in dict(Counter(test_y)).items()}
        }

    # print(f'Validation results: {val_res_dict}')
    # print(f'Validation classification report: {val_clf_report}')
    print(f'Test results: {test_res_dict}')
    print(f'Test classification report: {test_clf_report}')

    json.dump(test_res, open(f'results/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.json', 'w'), indent=4)
    open(f'results/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.txt', 'w').write(test_clf_report)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-ftr", "--ftr", help="feature type: numerical or textual", default="text", choices=["text", "num"])
    argParser.add_argument("-clf", "--clf", help="classification algorithm", default="rf", choices=["rf", "svm", "bert", "if"])
    argParser.add_argument("-do_down", "--do_down", help="do downsampling", action="store_true")
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])
    argParser.add_argument("-std", "--std", help="save data to disk", action="store_true")

    args = argParser.parse_args()
    print(args)

    main(args.ftr, args.clf, args.theme, args.do_down, args.std)
    print("Done!!!")

