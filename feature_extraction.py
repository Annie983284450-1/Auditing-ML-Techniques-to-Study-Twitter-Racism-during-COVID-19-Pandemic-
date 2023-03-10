import os
import sys
import re
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn import preprocessing
import string
 
 

def process_tweet(tweet):
    """
        Preprocess tweet. Remove URLs, leading user handles, retweet indicators, emojis,
        and unnecessary white space, and remove the pound sign from hashtags. Return preprocessed
        tweet in lowercase.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """

    # Remove www.* or https?://*
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))\s+', '', tweet)
    tweet = re.sub('\s+((www\.[^\s]+)|(https?://[^\s]+))', '', tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', tweet)
    # Remove RTs
    tweet = re.sub('^RT @[A-Za-z0-9_]+: ', '', tweet)
    # Incorrect apostraphe
    tweet = re.sub(r"’", "'", tweet)
    # Remove @username
    tweet = remove_leading_usernames(tweet)
    # Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Replace ampersands
    tweet = re.sub(r' &amp; ', ' and ', tweet)
    tweet = re.sub(r'&amp;', '&', tweet)
    # Remove emojis
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    # trim
    tweet = tweet.strip('\'"')
    return tweet.lower().strip()


def remove_leading_usernames(tweet):
    """
        Remove all user handles at the beginning of the tweet.

        Parameters
        -----------------
        tweet : str, a valid string representation of the tweet text
    """
    regex_str = '^[\s.]*@[A-Za-z0-9_]+\s+'

    original = tweet
    change = re.sub(regex_str, '', original)

    while original != change:
        original = change
        change = re.sub(regex_str, '', original)

    return change


def read_annotations(in_fn):
    df = pd.read_csv(in_fn, dtype=str)
    df_dict = df.to_dict('series')
    # word tokenize each tweet
    for i in range(len(df_dict['Text'])):
        df_dict['Text'][i] = process_tweet(df_dict['Text'][i])
        df_dict['Text'][i] = remove_leading_usernames(df_dict['Text'][i])
        df_dict['Text'][i] = list(word_tokenize(df_dict['Text'][i]))
    # each element in df_text is a list of words in this tweet
    # df_text = df_dict['Text']
    return df_dict


# k-top frequent words in the whole dataset
def tf_idf_K(df_dict, k):
    # df_text = df_dict['Text']
    all_words = []
    for i in range(len(df_dict['Text'])):
        all_words.extend(df_dict['Text'][i])
    print(all_words)
    print(len(all_words))
    all_words = nltk.FreqDist(all_words)
    topk = all_words.most_common(k)
    # print(topk)

    return topk, all_words

def find_features(document, all_words, k):
    words = set(document)
    features = {}
    word_features = list(all_words.keys())[:k]
    for w in word_features:
        features[w] = (w in words)
    return features


if __name__ == '__main__':
    usage = "Usage: python feature_extraction.py [input_file]!"
    if 2 != len(sys.argv):
        print(usage)
        sys.exit()
    in_fn = sys.argv[1]

##  preprocessing the tweets
    df_dict = read_annotations(in_fn)
 ## you guys can change the value of k, the bigger k, the more features you will get
 ## However, I have tried 100-1000, accuracy did not change that much
    k = 300
    topk, all_words = tf_idf_K(df_dict, k)
    feature_sets = []
    for i in range(len(df_dict['Text'])):
        feature_vec = find_features(df_dict['Text'][i], all_words, k)
        feature_label = df_dict['Label'][i]
        feature_sets.append((feature_vec,feature_label))
    # print(len(feature_sets))

    # the feature_sets contains feature vector (based on TF_IDF), 
    # and the corresponding label
    print(feature_sets)
    # split the data set into training and testing dataset
    train_set, test_set = feature_sets[100:len(df_dict['Text'])-1], feature_sets[:100]

# # predictions
# #     classifier = nltk.NaiveBayesClassifier.train(train_set)
# #     print("Naive Bayes accuracy percent:", nltk.classify.accuracy(classifier, test_set))

#     MNB_classifier = SklearnClassifier(MultinomialNB())
#     MNB_classifier.train(train_set)
#     print("MultinomialNB accuracy percent:", nltk.classify.accuracy(MNB_classifier, test_set))

#     # BNB_classifier = SklearnClassifier(BernoulliNB())
#     # BNB_classifier.train(train_set)
#     # print("BernoulliNB accuracy percent:", nltk.classify.accuracy(BNB_classifier, test_set))

#     LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#     LogisticRegression_classifier.train(train_set)
#     print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, test_set))*100)

#     SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#     SGDClassifier_classifier.train(train_set)
#     print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, test_set))*100)

#     SVC_classifier = SklearnClassifier(SVC())
#     SVC_classifier.train(train_set)
#     print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, test_set))*100)

#     LinearSVC_classifier = SklearnClassifier(LinearSVC())
#     LinearSVC_classifier.train(train_set)
#     print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, test_set))*100)

#     # NuSVC_classifier = SklearnClassifier(NuSVC())
#     # NuSVC_classifier.train(train_set)
#     # print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, test_set))*100)
