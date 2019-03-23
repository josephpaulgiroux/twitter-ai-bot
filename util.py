import os
import cPickle as pickle
import datetime as dt
import time
import os
import errno
import random
from nltk.tokenize.casual import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import numpy as np
import string
from pprint import pprint as pp
from collections import defaultdict
import sys
import sys
from kitchen.text.converters import getwriter
from bs4 import UnicodeDammit


UTF8Writer = getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)


PICKLE_PATH = os.path.expanduser("/home/joe/PickleJar/{}")

def make_sure_path_exists(path):
    try:
        path = os.path.expanduser(path)
        path = path.split("/")[:-1]
        path = "/" + ("/".join(path))
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise



def pickle_me(obj, filename):
    path = PICKLE_PATH.format(filename)
    make_sure_path_exists(path)
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)

def unpickle_me(filename):
    with open(PICKLE_PATH.format(filename), 'r') as fh:
        obj = pickle.load(fh)
    return obj



def str_to_date(date_str):
    return dt.datetime.strptime(date_str[:-10] + date_str[-4:], "%a %b %d %H:%M:%S %Y")





def stem_and_tokenize(text):
    stemmer = SnowballStemmer("english")
    sw = stopwords.words("english")
    word_stream = []
    tokens = tok.tokenize(text)
    num_unique = len(set(tokens))
    num_total = len(tokens)
    num_tags = 0
    num_content = 0
    for token in tokens:
        if token[0] in string.punctuation:
            num_tags += 1
        else:
            num_content += 1
    content_tag_ratio = float(num_content) / (num_tags + 1)

    for word in tokens:
        if "/" not in word  and "\\" not in word and len(word) > 1 and word not in sw and word[0] not in ["#", "@"]:
            word_stream.append(stemmer.stem(word))
    return num_unique, num_total, content_tag_ratio, " ".join(word_stream)





def train_test_split(features, labels, ratio=0.1):
    assert len(features) == len(labels)
    test_indices = np.random.choice(len(labels), int(len(labels)*ratio))
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    for n, label in enumerate(labels):
        if n in test_indices:
            features_test.append(features[n])
            labels_test.append(labels[n])
        else:
            features_train.append(features[n])
            labels_train.append(labels[n])

    print len(features_train)
    print len(features_test)
    print len(labels_train)
    print len(labels_test)
    return (features_train, features_test, labels_train, labels_test)


def fit_and_score(model, data):
    features_train, features_test, labels_train, labels_test = \
        (np.array(item) for item in data)
    model.fit(features_train, labels_train)
    # pred = model.predict(features_test)
    # acc = accuracy_score(pred, labels_test)
    pred = model.predict(features_test)
    acc = model.score(features_test, labels_test)
    print "train acc = {}".format(model.score(features_train, labels_train))
    print "test acc = {}".format(acc)
    return model, data, acc


tok = TweetTokenizer(preserve_case=False)
