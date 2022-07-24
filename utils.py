import time
import random
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
random.seed(42)


class SentenceEmbedder:
    def __init__(self):
        self.encoder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    def embed_texts(self, texts):
        return self.encoder.encode(texts)


class DataLoader:
    def __init__(self, downsample=False, theme='streak', ftr_type='text') -> None:
        self.theme = theme
        self.ftr_type = ftr_type
        self.downsample = downsample

        if self.theme == 'streak':
            self.num_cols = list(range(1, 6))
        elif self.theme == 'standing':
            self.num_cols = list(range(1, 17))
        elif self.theme == 'average':
            self.num_cols = list(range(1, 6))
        elif self.theme == 'double':
            self.num_cols = list(range(1, 8))

        if self.ftr_type == 'text':
            self.col_names = ['text', 'label']
        elif self.ftr_type == 'num':
            self.col_names = self.num_cols + ['label']
        elif self.ftr_type == 'emb':
            self.num_cols = list(range(1, 769))
            self.col_names = self.num_cols + ['label']

        self.train = pd.read_csv(f'data/{self.theme}/train_{self.ftr_type}{"_down" if self.downsample else ""}.csv', names=self.col_names)
        self.val = pd.read_csv(f'data/{self.theme}/validation_{self.ftr_type}{"_down" if self.downsample else ""}.csv', names=self.col_names)
        self.test = pd.read_csv(f'data/{self.theme}/test_{self.ftr_type}{"_down" if self.downsample else ""}.csv', names=self.col_names)
    
    def get_data(self):
        train_y = self.train['label'].to_numpy()
        val_y = self.val['label'].to_numpy()
        test_y = self.test['label'].to_numpy()
        if self.ftr_type == 'text':
            train_x = self.train['text'].to_list()
            val_x = self.val['text'].to_list()
            test_x = self.test['text'].to_list()
        elif self.ftr_type == 'num' or self.ftr_type == 'emb':
            train_x = self.train[self.num_cols].to_numpy()
            val_x = self.val[self.num_cols].to_numpy()
            test_x = self.test[self.num_cols].to_numpy()
        return train_x, train_y, val_x, val_y, test_x, test_y


class PrepData:
    def __init__(self, downsample=False, theme='streak', ftr_type='text') -> None:
        self.theme = theme
        self.ftr_type = ftr_type
        self.downsample = downsample

    def upsample_data(self, X, y):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

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

    def prep_data(self, all_data, embed_texts=False):

        train_x = np.array(all_data['train']['ftrs'])
        val_x = np.array(all_data['validation']['ftrs'])
        test_x = np.array(all_data['test']['ftrs'])
        train_y = np.array(all_data['train']['labs'])
        val_y = np.array(all_data['validation']['labs'])
        test_y = np.array(all_data['test']['labs'])

        if self.downsample:
            print('Downsampling...')
            train_x, train_y = self.downsample_data(train_x, train_y)

        if embed_texts:
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
        elif algo_name == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=5)
        elif algo_name == 'ann':
            self.clf = MLPClassifier(random_state=self.random_state)
        elif algo_name == 'lr':
            self.clf = LogisticRegression(random_state=self.random_state)

    def train(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, model, X):
        if self.algo_name == 'if':
            return np.array([1 if pred == -1 else 0 for pred in model.predict(X)])
        else:
            return model.predict(X)


class TpotThemeClassifier:
    def __init__(self, theme='streak', downsample=True) -> None:
        self.theme = theme
        self.down = downsample
    
    def train_streak_down(self, X, y):
        print('Training streak classifier...')
        from sklearn.ensemble import RandomForestClassifier
        exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=13, min_samples_split=17, n_estimators=100)
        # Fix random state in exported estimator
        if hasattr(exported_pipeline, 'random_state'):
            setattr(exported_pipeline, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train_standing_down(self, X, y):
        print('Training standing classifier...')
        from sklearn.cluster import FeatureAgglomeration
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures
        from tpot.builtins import StackingEstimator
        from tpot.export_utils import set_param_recursive
        exported_pipeline = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
            StackingEstimator(estimator=GaussianNB()),
            FeatureAgglomeration(affinity="manhattan", linkage="complete"),
            ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=18, min_samples_split=6, n_estimators=100)
        )
        # Fix random state for all the steps in exported pipeline
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train_streak(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.4, min_samples_leaf=18, min_samples_split=20, n_estimators=100)
        # Fix random state in exported estimator
        if hasattr(exported_pipeline, 'random_state'):
            setattr(exported_pipeline, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train_standing(self, X, y):
        from sklearn.decomposition import FastICA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from tpot.export_utils import set_param_recursive
        exported_pipeline = make_pipeline(
            FastICA(tol=0.35000000000000003),
            KNeighborsClassifier(n_neighbors=89, p=1, weights="uniform")
        )
        # Fix random state for all the steps in exported pipeline
        set_param_recursive(exported_pipeline.steps, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline
    
    def train_average_down(self, X, y):
        from sklearn.tree import DecisionTreeClassifier
        exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=3, min_samples_leaf=11, min_samples_split=6)
        # Fix random state in exported estimator
        if hasattr(exported_pipeline, 'random_state'):
            setattr(exported_pipeline, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train_double_down(self, X, y):
        from sklearn.ensemble import ExtraTreesClassifier
        exported_pipeline = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.1, min_samples_leaf=4, min_samples_split=14, n_estimators=100)
        if hasattr(exported_pipeline, 'random_state'):
            setattr(exported_pipeline, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train(self, X, y):
        if self.theme == 'streak' and self.down == True:
            return self.train_streak_down(X, y)
        elif self.theme == 'streak' and self.down == False:
            return self.train_streak(X, y)
        elif self.theme == 'standing' and self.down == True:
            return self.train_standing_down(X, y)
        elif self.theme == 'standing' and self.down == False:
            return self.train_standing(X, y)
        elif self.theme == 'average' and self.down == True:
            return self.train_average_down(X, y)
        elif self.theme == 'double' and self.down == True:
            return self.train_double_down(X, y)

    def predict(self, model, X):
        return model.predict(X)


class RuleClassifier:
    def __init__(self, theme='streak'):
        self.theme = theme

    def streak_rule(self, X):
        return np.array([1 if x[1] > 3 or x[3] > 3 else 0 for x in X])

    def standing_rule(self, X):
        pass

    def predict(self, model, X):
        if self.theme == 'streak':
            return self.streak_rule(X)
        elif self.theme == 'standing':
            return self.standing_rule(X)
