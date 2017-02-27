#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    27/02/2017
#
# *************************************** #


import os
import sys
from gensim import corpora, matutils, models
import logging
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from models.src.sentirueval import load_as_df
from models.src.classification import NLPClassification
from models.src.preprocessing import NLPPreprocessing


def main(args=None):
    print("Create Word2Vec model and perform classification.")


def sentence_metric(model, text):
    # Calculate vector for sentence
    featureVec = np.zeros((100,), dtype="float32")

    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in text:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec, model[word])

    # Divide the result by the number of words to get the average
    if nwords != 0:
        featureVec = np.divide(featureVec, nwords)

    return featureVec


def train_model(raw_corpus):
    # Pre-processing
    texts = NLPPreprocessing.preprocess_text(raw_corpus)
    processed_corpus = NLPPreprocessing.more_than_once(texts)

    # Create Word2Vec model
    w2v = models.Word2Vec(processed_corpus, size=100, alpha=0.025, window=5, min_count=1,
                          max_vocab_size=None, sample=1e-3, seed=1, workers=3, min_alpha=0.0001,
                          sg=1, hs=0, negative=5)

    w2v_corpus = [sentence_metric(w2v, text) for text in processed_corpus]

    return w2v_corpus, w2v


def apply_model(w2v, raw_corpus):
    # Pre-processing
    texts = NLPPreprocessing.preprocess_text(raw_corpus)
    processed_corpus = NLPPreprocessing.more_than_once(texts)

    # Apply model for new data
    w2v_corpus = [sentence_metric(w2v, text) for text in processed_corpus]

    return w2v_corpus


if __name__ == '__main__':
    # Show help message
    main()

    # Track w2v training process
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Load train and test data
    train_labels, raw_train_corpus = load_as_df(NLPClassification.FILE_TRAIN_BANK)
    test_labels, raw_test_corpus = load_as_df(NLPClassification.FILE_TEST_BANK)

    # Train model using train data
    train_vectors, w2v = train_model(raw_train_corpus)

    # Apply trained model
    test_vectors = apply_model(w2v, raw_test_corpus)

    classifier = NLPClassification()

    train_vectors_norm = MinMaxScaler().fit_transform(train_vectors)
    test_vectors_norm = MinMaxScaler().fit_transform(test_vectors)
    nb_classifier = classifier.train_nb_classifier(train_vectors_norm, train_labels)
    classifier.test_classifier(nb_classifier, test_vectors_norm, test_labels)

    nb_classifier = classifier.train_nb2_classifier(train_vectors, train_labels)
    classifier.test_classifier(nb_classifier, test_vectors, test_labels)

    classifier.test_all_classifiers(train_vectors, train_labels,
                                    test_vectors, test_labels)
