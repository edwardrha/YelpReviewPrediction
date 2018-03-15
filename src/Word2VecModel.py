# -*- coding: utf-8 -*-
from gensim.models import Doc2Vec
import gensim
import pandas as pd
from multiprocessing import cpu_count, Pool


class LabeledLineSentence(object):
    def __init__(self, articles, labels):
        self.articles = articles
        self.labels = labels
    def __iter__(self):
        for i in range(len(self.articles)):
            yield gensim.models.doc2vec.TaggedDocument(words=self.articles[i], tags=[self.labels[i]])
    def __len__(self):
        return len(self.labels)


def Bootstrap():
    sizes = [5, 10, 25, 50]
    df = pd.read_json('../dataset/processed.json')
    for i in range(len(sizes)):
        tag_stream = LabeledLineSentence(df['text'], df['review_id'])
        model = Doc2Vec(tag_stream, alpha=0.1, min_alpha=0.005, window=10, size=sizes[i], iter=20, min_count=8, workers=cpu_count()*4)
        save(model, path = '../models/word2vec_model'+str(sizes[i]))


def train(model, data, labels):
    tag_stream = LabeledLineSentence(list(data), list(labels))
    model.train(tag_stream, total_examples=model.corpus_count, epochs=model.iter)
    return model


def save(model, path = 'models/word2vec_model10'):
    model.save(path)


def load(path = 'models/word2vec_model10'):
    model = Doc2Vec.load(path)
    return model


def vectorizeReview(model, datapoint):
    output = model.infer_vector(datapoint)
    return output


def vectorizeWord(model, word):
    return model[word]


# if __name__ == '__main__':
    # Bootstrap()
