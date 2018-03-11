# -*- coding: utf-8 -*-
if __name__ == "__main__":
    from gensim.models import Doc2Vec
    import gensim
import pandas as pd
from multiprocessing import cpu_count, Pool
import string

class LabeledLineSentence(object):
    def __init__(self, articles, labels):
        # bigram = gensim.models.phrases.Phraser.load('../models/phraser')
        # self.articles = list(bigram[articles])
        self.articles = articles
        self.labels = labels

    def __iter__(self):
        for i in range(len(self.articles)):
            yield gensim.models.doc2vec.TaggedDocument(words=self.articles[i], tags=[self.labels[i]])
    def __len__(self):
        return len(self.labels)

def Bootstrap():
    # sizes = [5, 10, 25, 50, 100]
    # for i in range(5):
    #     model = Doc2Vec(alpha=0.1, min_alpha=0.005, window=8, size=sizes[i], iter=20, min_count=10, workers=cpu_count())
    #     for i in range(1,7):
    #         df = pd.read_json('../dataset/processed_'+str(i)+'.json')
    #         model = train(model, df['text'], df['review_id'])
    #     save(model, path = '../models/word2vec_model'+str(sizes[i]))
    model = Doc2Vec(alpha=0.1, min_alpha=0.005, window=8, size=10, iter=20, min_count=10, workers=cpu_count())
    

def train(model, data, labels):
    tag_stream = LabeledLineSentence(list(data), list(labels))
    model.build_vocab(tag_stream)
    tag_stream = LabeledLineSentence(list(data), list(labels))
    model.train(tag_stream, total_examples=model.corpus_count, epochs=model.iter)
    return model

def save(model, path = '../models/word2vec_model10'):
    model.save(path)

def load(model, path = '../models/word2vec_model10'):
    model = Doc2Vec.load(path)
    return model

def vectorizeReview(model, data):
    output = model.infer_vector(data)
    return output

def vectorizeWord(model, word):
    return model[word]

if __name__ == '__main__':
    Bootstrap()
