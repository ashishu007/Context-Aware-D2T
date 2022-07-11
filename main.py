import numpy as np
import pandas as pd
import argparse, json, pickle
from collections import Counter
from team_themes import TeamStanding, TeamStreak
from bert_utils import BertDataLoader, BertThemeClassifier
from utils import DataLoader, Classifier, TpotThemeClassifier
from transformers import AutoModelForSequenceClassification
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
    train = pd.DataFrame({"text": train_x, "label": list(train_y)})
    val = pd.DataFrame({"text": val_x, "label": list(val_y)})
    test = pd.DataFrame({"text": test_x, "label": list(test_y)})
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    val = val.sample(frac=1, random_state=42).reset_index(drop=True)
    test = test.sample(frac=1, random_state=42).reset_index(drop=True)
    train.to_csv(f'data/{THEME}/train_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    val.to_csv(f'data/{THEME}/validation_text{"_down" if DOWN else ""}.csv', index=False, header=False)
    test.to_csv(f'data/{THEME}/test_text{"_down" if DOWN else ""}.csv', index=False, header=False)


def main(FTR='text', CLF='rf', THEME='streak', DOWNSAMPLE=False, SAVE_TO_DISK=False, AUGMENT=False):

    if CLF == 'if':
        assert DOWNSAMPLE == False, 'Downsampling not supported for Isolation Forest as it is an anomaly detection algorithm'
    elif CLF == 'bert' or CLF == 'pet':
        assert FTR == 'text', f'{CLF.upper()} only works with text features'

    if THEME == 'streak':
        theme_obj = TeamStreak(FTR)
    elif THEME == 'standing':
        theme_obj = TeamStanding(FTR)

    print('Processing data...')
    all_data, aug_texts = theme_obj.get_theme_train_val_test_data()

    if AUGMENT:
        print(f'Saving {len(aug_texts)} augmented data!!!')
        if FTR == 'text':
            pd.DataFrame({"text": aug_texts, 'labels': [0]*len(aug_texts)}).to_csv(f'data/{THEME}/augmented_text.csv', index=False, header=False)
        elif FTR == 'num':
            # pd.DataFrame({"num": aug_texts, 'labels': [0]*len(aug_texts)}).to_csv(f'data/{THEME}/augmented_num.csv', index=False, header=False)
            pass
        return

    if CLF == 'bert' or CLF == 'pet':
        clf_obj = BertThemeClassifier()
        dl_obj = BertDataLoader(downsample=DOWNSAMPLE)

        print('Preprocessing data...')
        dataset, train_x, train_y, val_x, val_y, test_x, test_y = dl_obj.prep_data(all_data)
        print(f"Train: {len(train_x)} {train_y.shape}\tVal: {len(val_x)} {val_y.shape}\tTest: {len(test_x)} {test_y.shape}")
        print(Counter(train_y), Counter(val_y), Counter(test_y))

        if SAVE_TO_DISK:
            save_to_disk(train_x, train_y, val_x, val_y, test_x, test_y, THEME, DOWN=DOWNSAMPLE)
            return 

        if CLF == 'bert':
            print('Training BERT model...')
            model, tokenizer = clf_obj.train(dataset)
            model_path = f"models/{THEME}/{CLF}-down_{'yes' if DOWNSAMPLE else 'no'}"
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

    else:
        if CLF == 'tpot':
            print('This is using tpot model...')
            assert FTR == 'num', 'TPOT only works with numerical features'
            clf_obj = TpotThemeClassifier(theme=THEME, downsample=DOWNSAMPLE)
            dl_obj = DataLoader(ftr_type=FTR, downsample=DOWNSAMPLE)
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
        trained_model = pickle.load(open(f'models/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.pkl', 'rb'))
        test_pred = clf_obj.predict(trained_model, test_x)

    np.save(open(f'preds/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.npy', 'wb'), test_pred)
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

    json.dump(test_res, open(f'results/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.json', 'w'), indent=4)
    open(f'results/{THEME}/{CLF}-{FTR}-down_{"yes" if DOWNSAMPLE else "no"}.txt', 'w').write(test_clf_report)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-std", "--std", help="save data to disk", action="store_true")
    argParser.add_argument("-down", "--down", help="do downsampling", action="store_true")
    argParser.add_argument("-aug", "--aug", help="augment data for pet training", action="store_true")
    argParser.add_argument("-ftr", "--ftr", help="feature type: numerical or textual", default="text", choices=["text", "num"])
    argParser.add_argument("-clf", "--clf", help="classification algorithm", default="rf", \
                            choices=["rf", "svm", "bert", "if", "pet", 'tpot'])
    argParser.add_argument("-theme", "--theme", help="across-event theme", default='standing', \
                            choices=['streak', 'standing', 'double', 'average'])

    args = argParser.parse_args()
    print(args)

    main(args.ftr, args.clf, args.theme, args.down, args.std, args.aug)
    print("Done!!!")

