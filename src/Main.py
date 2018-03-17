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

import numpy as np
import pandas as pd
import csv
import time

df = pd.read_json('dataset/processed_restaurant_reviews_1.json')
data = df.text.apply(' '.join)[:300000]



sizes = [10, 15, 20, 25, 30, 40]
i = 3 # Change this to open labels for different cluster size
thefile = open('dataset/300ktop_category_' + str(sizes[i]) + '.txt', 'r')
reader = csv.reader(thefile)
labels = [] # This will store the labels
for row in reader:
    labels.append(row[0])
thefile.close()
labels = np.array(labels)

for j in range(sizes[i]):
    print "Topic %d:" % (j)
    for item in  data[labels==str(j)][:20]:
        print item, '\n'
    print "\n"
