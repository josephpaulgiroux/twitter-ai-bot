# -*- coding: UTF-8 -*-

import numpy as np
from collections import defaultdict
from markov_const import *
from itertools import izip
import string
from bs4 import UnicodeDammit

from nltk.tokenize.casual import TweetTokenizer, remove_handles, reduce_lengthening, casual_tokenize
import regex as re
re.default_version = re.VERSION1

class WeightedMarkovBabbler(object):

    def __init__(self, n_back=1):
        self.n_back = n_back
        self.chain = dict()

    def ingest(self, document, weight=1.0, purge_list=PURGE_LIST):
        doc_lower = UnicodeDammit(document).unicode_markup.lower()
        for word in BLACKLIST:
            if word in doc_lower:
                return
        word_stream = [item for item in casual_tokenize(document, reduce_len=True) if "/" not in item and item not in purge_list]
        if len(word_stream) < 3:
            return
        word_stream = [BEGIN_SYMBOL] * self.n_back + word_stream
        word_stream += [END_SYMBOL] * self.n_back
        for current_position, word in enumerate(word_stream):
            if current_position < self.n_back:
                # advance until we have enough words in memory to consider
                # (even if only begin symbols)
                continue
            prior_ngram = word_stream[(current_position - self.n_back):current_position]
            result_word = word_stream[current_position]
            self.chain = record_chain_link(prior_ngram, result_word, self.chain, weight)

    def fit(self, documents, weights=None):
        if weights is not None:
            for document, weight in izip(documents, weights):
                self.ingest(document, weight)
        else:
            for document in documents:
                self.ingest(document)

    def babble(self, max_len=140, require=None):
        raw_stream = [BEGIN_SYMBOL] * self.n_back
        total_len = 0
        done = False
        hashtags = []
        def is_content_word(word):
            return 0 if word[0] in string.punctuation else 1
        if require:
            max_len -= len(require)

        num_content = 0
        while not done:
            new_word, all_possible = advance_from_ngram(raw_stream[-self.n_back:], self.chain, require=require, forbid=hashtags)
            if new_word == require:
                hashtags.append(require.lower())
                max_len += len(require)
                require = None
            elif new_word[0] in ["#", "@"]:
                lower_tag = new_word.lower()
                if lower_tag in hashtags:
                    new_word = END_SYMBOL
                else:
                    hashtags.append(lower_tag)

            if total_len + 1 + len(new_word) > max_len:
                if END_SYMBOL in all_possible:
                    new_word = END_SYMBOL
                else:
                    return None, None
            if new_word == END_SYMBOL:
                done = True

            num_content += (is_content_word(new_word) and new_word not in raw_stream)
            raw_stream.append(new_word)
            total_len += 1 + len(new_word)

        word_stream = [word for word in raw_stream if word not in [BEGIN_SYMBOL, END_SYMBOL]]
        if require and (require not in word_stream) and (require.lower() not in hashtags):
            if np.random.random() < 0.5:
                print "Adding to beginning"
                word_stream = [require] + word_stream
            else:
                print "Adding to end"
                word_stream.append(require)

        clean_text = " ".join(word_stream)
        clean_text = clean_text.lstrip(string.punctuation.replace("#", "").replace("@", "") + " ")
        for regex, replacement in CLEANUP_REGEXES:
            clean_text = re.sub(regex, replacement, clean_text)
        print "Cleaned: ", clean_text
        return num_content, clean_text

def advance_from_ngram(ngram, chain, forbid=[], require=None, require_chance=0.30):
    if len(ngram) == 1:
        prior_word = ngram[0]

        choices = chain[prior_word]
        valid_choices = {k:v for k,v in choices.items() if k not in forbid}
        if not valid_choices:
            print "No valid choices left."
            return END_SYMBOL

        if require and (require in valid_choices):
            if np.random.random() < require_chance:
                print "Keeping required tag", require
                return require
            else:
                print "Not keeping tag."

        weight_sum = np.sum(valid_choices.values())
        # for item in choices.itervalues():
        #     print item
        weights = [item/weight_sum for item in valid_choices.itervalues()]
        # print "Keys are", choices.keys()

        choice = np.random.choice(valid_choices.keys(), p=weights)
        return choice, valid_choices.keys()
    else:
        sub_ngram = ngram[1:]
        subchain = chain[ngram[0]]
        return advance_from_ngram(sub_ngram, subchain)

    return END_SYMBOL



def record_chain_link(ngram, result_word, chain, weight=1.0):
    # print "recording at ngram length {}".format(len(ngram))
    if len(ngram) == 1:
        prior_word = ngram[0]
        if prior_word in chain:
            if result_word in chain[prior_word]:
                # print u"Found prior word {}".format(prior_word)
                chain[prior_word][result_word] += weight
            else:
                # print u"Did not find prior word {} case 1".format(prior_word)
                chain[prior_word][result_word] = weight
        else:
            # print u"Did not find prior word {} case 2".format(prior_word)
            chain[prior_word] = {result_word: weight}
        return chain
    else:
        first_word = ngram[0]
        sub_ngram = ngram[1:]
        if first_word in chain:
            # print u"Found {} already".format(first_word)
            subchain = chain[first_word]
        else:
            # print u"Did not find first word {}".format(first_word)
            subchain = {}

        chain[first_word] = record_chain_link(sub_ngram, result_word, subchain, weight)
        return chain

