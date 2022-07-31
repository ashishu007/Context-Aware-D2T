import json
import random
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import NearMiss, OneSidedSelection, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, SMOTEN
from imblearn.under_sampling import RandomUnderSampler, AllKNN, RepeatedEditedNearestNeighbours, InstanceHardnessThreshold

random.seed(42)


class DataLoader:
    def __init__(self, theme='streak', season='all'):
        self.theme = theme
        self.season = season
        self.all_data = json.load(open(f'data/{self.theme}/all_data.json'))
        self.seasonal_splits = json.load(open(f'data/seasonal_splits.json'))
        self.set_splits = self.seasonal_splits[season]

    def get_data(self):
        data = {}
        for part in ['train', 'validation', 'test']:
            ftrs, labs = [], []
            data[part] = {}
            part_gem_ids = self.set_splits[part]
            for key, val in self.all_data.items():
                if key in part_gem_ids:
                    for item in val:
                        ftrs.append(list(item['ftrs'].values()))
                        labs.append(item['lab'])
            data[part]['ftrs'] = np.array(ftrs)
            data[part]['labs'] = np.array(labs)
        return np.array(data['train']['ftrs']), np.array(data['train']['labs']), \
                np.array(data['validation']['ftrs']), np.array(data['validation']['labs']), \
                np.array(data['test']['ftrs']), np.array(data['test']['labs'])


class Classifier:
    def __init__(self, algo_name='rf'):
        self.random_state = 42
        self.algo_name = algo_name
        if algo_name == 'rf':
            self.clf = RandomForestClassifier(random_state=self.random_state)
        elif algo_name == 'svm':
            self.clf = SVC(random_state=self.random_state, kernel='linear', probability=True)
        elif algo_name == 'knn':
            self.clf = KNeighborsClassifier(n_neighbors=5)
        elif algo_name == 'ann':
            self.clf = MLPClassifier(random_state=self.random_state)
        elif algo_name == 'lr':
            self.clf = LogisticRegression(random_state=self.random_state)

    def train(self, X, y):
        return self.clf.fit(X, y)

    def predict(self, model, X):
        return model.predict(X)
    
    def predict_proba(self, model, X):
        return model.predict_proba(X)


class Sampler:
    def __init__(self, method='smote'):
        self.random_state = 42
        self.method = method
        if self.method == 'smote':
            self.sampler = SMOTE(random_state=self.random_state)
        elif method == 'adasyn':
            self.sampler = ADASYN(random_state=self.random_state)
        elif method == 'svmsmote':
            self.sampler = SVMSMOTE(random_state=self.random_state)
        elif method == 'smoten':
            self.sampler = SMOTEN(random_state=self.random_state)
        elif method == 'borderlinesmote':
            self.sampler = BorderlineSMOTE(random_state=self.random_state)
        elif method == 'kmeanssmote':
            self.sampler = KMeansSMOTE(random_state=self.random_state)
        elif method == 'randomoversampler':
            self.sampler = RandomOverSampler(random_state=self.random_state)
        elif method == 'randomundersampler':
            self.sampler = RandomUnderSampler(random_state=self.random_state)
        elif method == 'allknn':
            self.sampler = AllKNN()
        elif method == 'repeatededitednearestneighbours':
            self.sampler = RepeatedEditedNearestNeighbours()
        elif method == 'instancehardnessthreshold':
            self.sampler = InstanceHardnessThreshold(random_state=self.random_state)
        elif method == 'nearmiss':
            self.sampler = NearMiss()
        elif method == 'onesidedselection':
            self.sampler = OneSidedSelection(random_state=self.random_state)
        elif method == 'tomeklinks':
            self.sampler = TomekLinks()
        elif method == 'editednearestneighbours':
            self.sampler = EditedNearestNeighbours()

    def sample_data(self, X, y):
        if self.method == 'none':
            return X, y, 'success'
        else:
            try:
                X_res, y_res = self.sampler.fit_resample(X, y)
                msg = "success"
            except:
                X_res, y_res = X, y
                msg = "failure"
            return X_res, y_res, msg

