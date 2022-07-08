import random, time
import numpy as np
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, classification_report
random.seed(42)


class SentenceEmbedder:
    def __init__(self):
        self.encoder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    def embed_texts(self, texts):
        return self.encoder.encode(texts)


class DataLoader:
    def __init__(self, ftr_type='text', downsample=False):
        self.ftr_type = ftr_type
        self.downsample = downsample

    def downsample_data(self, X, y):
        pos_x = [X[idx] for idx, item in enumerate(y) if y[idx] == 1]
        pos_y = [y[idx] for idx, item in enumerate(y) if y[idx] == 1]
        neg_x = [X[idx] for idx, item in enumerate(y) if y[idx] == 0]
        neg_y = [y[idx] for idx, item in enumerate(y) if y[idx] == 0]
        neg_x1 = np.array(random.sample(neg_x, len(pos_x)))
        neg_y1 = np.array(random.sample(neg_y, len(pos_y)))
        train_x1 = np.concatenate([pos_x, neg_x1])
        train_y1 = np.concatenate([pos_y, neg_y1])
        return train_x1, train_y1

    def prep_data(self, all_data):

        train_x = np.array(all_data['train']['ftrs'])
        val_x = np.array(all_data['validation']['ftrs'])
        test_x = np.array(all_data['test']['ftrs'])
        train_y = np.array(all_data['train']['labs'])
        val_y = np.array(all_data['validation']['labs'])
        test_y = np.array(all_data['test']['labs'])

        if self.downsample:
            print('Downsampling...')
            train_x, train_y = self.downsample_data(train_x, train_y)

        if self.ftr_type == 'text':
            embed_obj = SentenceEmbedder()
            print('Embedding train texts...')
            t1 = time.time()
            train_x = embed_obj.embed_texts(train_x)
            t2 = time.time()
            print(f'Time taken to embed train texts: {t2-t1}')
            
            print('Embedding val texts...')
            t1 = time.time()
            val_x = embed_obj.embed_texts(val_x)
            t2 = time.time()
            print(f'Time taken to embed val texts: {t2-t1}')

            print('Embedding test texts...')
            t1 = time.time()
            test_x = embed_obj.embed_texts(test_x)
            t2 = time.time()
            print(f'Time taken to embed test texts: {t2-t1}')

        return train_x, train_y, val_x, val_y, test_x, test_y


class CalcPerformance:
    def __init__(self) -> None:
        pass

    def get_clf_report(self, true, pred):
        mf1 = float(f"{f1_score(true, pred, average='macro')*100:.2f}")    
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


class Classifier:
    def __init__(self, algo_name='rf'):
        self.random_state = 42
        self.algo_name = algo_name
        if algo_name == 'rf':
            self.clf = RandomForestClassifier(random_state=self.random_state)
        elif algo_name == 'svm':
            self.clf = LinearSVC(random_state=self.random_state)
        elif algo_name == 'if':
            self.clf = IsolationForest(random_state=self.random_state)

    def train(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, model, X):
        if self.algo_name == 'if':
            return np.array([1 if pred == -1 else 0 for pred in model.predict(X)])
        else:
            return model.predict(X)


class FeatureExtractor:
    def __init__(self):
        self.encoder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    def embed_texts(self, texts):
        return self.encoder.encode(texts)

    def team_streak_text_ftr(self, streak, broken_streak, team_name):
        """
        text: Philladelphia 76ers have won 1 consecutive matches. They broke 5 matches loosing streak.
        """
        text = f"{team_name} have {'won' if streak['type'] == 'W' else 'lost'} {streak['count']} consecutive matches."
        if broken_streak:
            text += f" They broke {broken_streak['count']} matches {'loosing' if broken_streak['type'] == 'L' else 'winning'} streak."
        return text

    def team_streak_num_ftr(self, streak, broken_streak):
        ids = {'W': 1, 'L': 2}
        ftrs = [streak['count'], ids[streak['type']]]
        if 'type' in broken_streak:
            ftrs.append(ids[broken_streak['type']])
        else:
            ftrs.append(0)
        if 'count' in broken_streak:
            ftrs.append(broken_streak['count'])
        else:
            ftrs.append(0)
        return ftrs

    def team_standing_text_ftr(self, info):
        """
        current_text: Boston Celtics are at place 4 in the standings with 10 wins, 5 losses and win percent of 66%. 
        prev_1_text: The team above them is Brooklyn Nets at place 3 in the stadings with 11 wins, 5 losses and win percent of 68%.
        next_1_text: The team below them is Toronto Raptors at place 5 in the standing with 9 wins, 5 losses and win percent of 64%.
        season_date_text: This is the 49 day of the season.
        """

        current_text = f'{info["current"]["team"]} are at place {info["current"]["standing"]} in the standings with {info["current"]["wins"]} wins, {info["current"]["losses"]} losses and win percent of {info["current"]["win_perct"]}%.'
        season_date_text = f'This is the {info["current"]["season_date"]} day of the season.'

        if 'prev_1' in info:
            prev_1_text = f'The team above them is {info["prev_1"]["team"]} at place {info["prev_1"]["standing"]} in the stadings with {info["prev_1"]["wins"]} wins, {info["prev_1"]["losses"]} losses and win percent of {info["prev_1"]["win_perct"]}%.'
        else:
            prev_1_text = 'There is no team above them.'

        if 'next_1' in info:
            next_1_text = f'The team below them is {info["next_1"]["team"]} at place {info["next_1"]["standing"]} in the standing with {info["next_1"]["wins"]} wins, {info["next_1"]["losses"]} losses and win percent of {info["next_1"]["win_perct"]}%.'
        else:
            next_1_text = 'There is no team below them.'

        return f'{current_text} {prev_1_text} {next_1_text} {season_date_text}'

    def team_standing_num_ftr(self, info):
        ftr_vector = {}
        current_team_ftrs = ['standing', 'wins', 'losses', 'win_perct', 'season_date']
        other_team_ftrs = ['standing', 'wins', 'losses', 'win_perct', 'win_diff']

        for key, val in info['current'].items():
            if key in current_team_ftrs:
                ftr_vector[f"current-{key}"] = val
        
        if 'prev_1' in info:
            for key in other_team_ftrs:
                ftr_vector[f"prev_1-{key}"] = info['prev_1'][key]
        else:
            for key in other_team_ftrs:
                ftr_vector[f"prev_1-{key}"] = 0
        
        if 'next_1' in info:
            for key in other_team_ftrs:
                ftr_vector[f"next_1-{key}"] = info['next_1'][key]
        else:
            for key in other_team_ftrs:
                ftr_vector[f"next_1-{key}"] = 0
        
        return list(ftr_vector.values())

    