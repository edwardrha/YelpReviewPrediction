# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import csv
from multiprocessing import Pool, cpu_count
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


def Bootstrap():
    df = pd.read_json('../dataset/processed_restaurant_reviews_1.json')
    data = df.text.apply(' '.join)[:300000]
    del(df)
    vec = CountVectorizer(min_df=5)
    X = vec.fit_transform(data)
    vocab = vec.get_feature_names()
    tf_feature_names = vec.get_feature_names()
    save(tf_feature_names, '../models/tf_feature_names.pkl')
    save(vec, '..models/countvectorizer.pkl')

    input = []
    sizes = [10, 15, 20, 25, 30, 40]
    for size in [10, 15, 20, 25, 30, 40]:
        input.append([X, size])
    p = Pool(cpu_count())
    results = p.map(createLDAmodel, input)
    p.close()
    p.join()
    for i in range(6):
        print "1\n"
        lda = results[i]
        predictions = lda.transform(X)
        print "2\n"
        results.append(lda)
        myFile = open('../dataset/300klabel_proba_' + str(sizes[i]) + '.txt', 'w')
        thefile = open('../dataset/300ktop_category_' + str(sizes[i]) + '.txt', 'w')
        with myFile:
            writer = csv.writer(myFile)
            writer.writerows(predictions)
            for item in predictions:
                thefile.write("%s\n" % np.argmax(item))
        thefile.close()
        myFile.close()
        save(lda, '../models/lda_'+str(sizes[i])+'.pkl')


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])
        print ""


def createLDAmodel(list1):
    X = list1[0]
    n_topics = list1[1]
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(X)
    return lda


def save(model, path='../models/lda.pkl'):
    joblib.dump(model, path)


def load(path='../models/lda.pkl'):
    model = joblib.load(path)
    return model


if __name__ == '__main__':
    Bootstrap()
