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

