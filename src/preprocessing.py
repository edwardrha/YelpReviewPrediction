#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from multiprocessing import cpu_count, Pool
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import unicodedata


# NOTE: MINIMUM 64GB OF RAM REQUIRED TO RUN BOOTSTRAP



# Global
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def Preprocessing(InputString, regexflag=0):
    """
        Input: String
        Output: List of words
    Takes in a string and removes stop words then lemmatizes the rest.
    """
    if regexflag == 1:
        InputString = re.sub('[%s]' % re.escape('["#%\'()*+,/:;<=>@\[\]^_`{|}~’”“′‘\\\.!?『』]+'), ' ', InputString)
    InputString = unicodedata.normalize('NFKD', InputString).encode('ascii','ignore')
    InputString = InputString.lower()
    word_tokens = word_tokenize(InputString)
    filtered_tokens = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]
    return filtered_tokens


def Bootstrap():
    """
    Bootstrap code to create processed data file.
    """
    df_temp = pd.read_json('../dataset/review.json', lines=True)
    df_temp.text = df_temp.text.str.replace('["#%\'()*+,/:;<=>@\[\]^_`{|}~’”“′‘\\\.!?『』]+', ' ')
    df_bus = pd.read_json('../dataset/business.json', lines=True)
    df_bus = df_bus[[('Restaurants' in categories) for categories in df_bus['categories']]]
    df_temp = pd.merge(df_temp, df_bus, how='inner', on='business_id')[['business_id', 'date', 'review_id', 'stars_x', 'text', 'user_id']]
    df_temp = df_temp.rename(index=str, columns={"stars_x": "stars"})
    p = Pool(4) # Increase if enough memory with >64GB
    df_temp.text = p.map(Preprocessing, df_temp.text)
    p.close()
    p.join()
    print "processed\n"
    # df_temp.to_json('../dataset/processed_restaurant_reviews.json')
    df_temp[:700000].to_json('../dataset/processed_restaurant_reviews_1.json')
    df_temp[700000:1400000].to_json('../dataset/processed_restaurant_reviews_2.json')
    df_temp[1400000:2100000].to_json('../dataset/processed_restaurant_reviews_3.json')
    df_temp[2100000:2800000].to_json('../dataset/processed_restaurant_reviews_4.json')
    df_temp[2800000:3500000].to_json('../dataset/processed_restaurant_reviews_5.json')


if __name__ == "__main__":
    Bootstrap()
