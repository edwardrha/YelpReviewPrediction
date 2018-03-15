import pandas as pd
import pickle
from sklearn.cluster import KMeans
from multiprocessing import cpu_count


def CreateModel(n_cluster = 10, n_init = 15):
    kMeansModel = KMeans(n_clusters = n_cluster, n_init = n_init, n_jobs=1, random_state=3425)
    return kMeansModel


def train(model, data):
    model.fit(data)
    return model


def save(model, path='/models/kMeansModel'):
    save_file = open(path, 'wb')
    test_str = pickle.dump(model, save_file, -1)


def load(path='/models/kMeansModel'):
    load_file = open(path, 'rb')
    output = pickle.load(load_file)


def predict(model, data):
    return model.predict(data)
