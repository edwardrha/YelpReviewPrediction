# -*- coding: utf-8 -*-
from preprocessing import Preprocessing
import Word2VecModel
import ClusterModel

import numpy as np
import pandas as pd
import csv
from multiprocessing import cpu_count
import string
from collections import Counter


def printReviewCluster(df, labels, target, size=50):
    counter = 0
    for i in df.loc[labels==target, 'text']:
        print string.join(i), '\n'
        counter += 1
        if counter == size:
            break;

def Make_word_cloud(input_data, cluster_labels):
    '''
        Input:  List reviews, labels
        Output: None
    ----------------------------------------------------------------------------
    Creates and saves a wordcloud for each cluster at '../image_outputs'
    '''
    import pytagcloud
    labels = np.unique(cluster_labels)
    for label in labels:
        content_list = []
        for item in input_data[cluster_labels==label]:
            content_list.append(item)
        split_to_words = ' '.join(content_list).split()
        counter = Counter(split_to_words)
        tags = counter.most_common(50)
        taglist = pytagcloud.make_tags(tags, maxsize=80)
        pytagcloud.create_tag_image(taglist, 'images/wordcloud_'+str(label)+'.png', size=(1000, 1000), rectangular=True, layout=4)

# model1 = Word2VecModel.load('../models/word2vec_model50')
# model1.wv.most_similar(positive=['burger', 'fast'], negative=['coffee'])
# model1.wv.most_similar(positive=['ramen'])
# model1.wv.most_similar(positive=['sushi', 'chinese'], negative=['japanese'])
# model1.wv.most_similar(['paris', 'london'], ['france'])
#
# df = pd.read_json('../dataset/processed_restaurant_reviews_1.json')
#
# df = df.loc[:300000]
