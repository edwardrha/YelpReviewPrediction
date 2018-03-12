# -*- coding: utf-8 -*-
import pandas as pd
from multiprocessing import cpu_count, Pool
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Global
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def Preprocessing(InputString):
    """
        Input: String
        Output: List of words
    Takes in a string and removes stop words then lemmatizes the rest.
    """
    # InputString = unicodedata.normalize('NFKD', InputString).encode('ascii','ignore')
    InputString = InputString.lower()
    word_tokens = word_tokenize(InputString)
    filtered_tokens = [lemmatizer.lemmatize(w) for w in word_tokens if not w in stop_words]
    return filtered_tokens

def Bootstrap():
    """
    Bootstrap code to create processed data files.
    """
    df_temp = pd.read_json('../dataset/review.json', lines=True)
    df_temp.text = df_temp.text.str.replace('["#%\'()*+,/:;<=>@\[\]^_`{|}~’”“′‘\\\.!?『』 ]+', ' ')
    p = Pool(2) # Increase if enough memory with >64GB
    df_temp.text = p.map(Preprocessing, df_temp.text)
    p.close()
    p.join()
    df_temp.to_json('../dataset/processed.json')
    # for i in range(1,7):
    #     print i
    #     df_temp = pd.read_json('../dataset/review_' + str(i) + '.json')
    #     df_temp.text = df_temp.text.str.replace('["#%\'()*+,/:;<=>@\[\]^_`{|}~’”“′‘\\\.!?『』 ]+', ' ')
    #     p = Pool(cpu_count())
    #     df_temp.text = p.map(Preprocessing, df_temp.text)
    #     p.close()
    #     p.join()
    #     df_temp.to_json('../dataset/processed_'+str(i)+'.json')


if __name__ == "__main__":
    Bootstrap()
