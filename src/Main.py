# -*- coding: utf-8 -*-
from preprocessing import Preprocessing
import Word2VecModel
import ClusterModel
import pandas as pd
from sklearn.cluster import KMeans
from multiprocessing import cpu_count


def createAndTrainCluster():
    model1 = Word2VecModel.load('../models/word2vec_model10')
    df = pd.read_json('../dataset/processed_6.json',).reset_index()
    vectors = []
    labels = []
    for i in xrange(df.shape[0]):
        vectors.append(model1.infer_vector(df.loc[i,'text']))
        labels.append(df.loc[i,'review_id'])
        if i%10000 == 0:
            print i
    del(df)
    return vectors, labels


vectors, labels = createAndTrainCluster()

model2 = ClusterModel.CreateModel()
model2 = ClusterModel.train(model2, vectors)
model2.labels_

model1.wv.most_similar(positive=['burger', 'fast'], negative=['coffee'])
model1.wv.most_similar(positive=['ramen', 'mexican'], negative=['japanese'])
model1.wv.most_similar(['paris', 'england'], ['london'])
