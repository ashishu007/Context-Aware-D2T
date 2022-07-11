import time
import random
import numpy as np
from sklearn.svm import LinearSVC
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
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


class TpotThemeClassifier:
    def __init__(self, theme='streak') -> None:
        self.theme = theme
    
    def train_streak(self, X, y):
        print('Training streak classifier...')
        from sklearn.ensemble import RandomForestClassifier

        exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.45, min_samples_leaf=13, min_samples_split=17, n_estimators=100)
        # Fix random state in exported estimator
        if hasattr(exported_pipeline, 'random_state'):
            setattr(exported_pipeline, 'random_state', 42)
        exported_pipeline.fit(X, y)
        return exported_pipeline

    def train_standing(self, X, y):
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

    def train(self, X, y):
        if self.theme == 'streak':
            return self.train_streak(X, y)
        elif self.theme == 'standing':
            return self.train_standing(X, y)

    def predict(self, model, X):
        return model.predict(X)
