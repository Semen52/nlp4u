#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    27/02/2017
#
# *************************************** #


from gensim import corpora, matutils
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'../../'))
from models.src.sentirueval import load_as_df
from models.src.classification import NLPClassification
from models.src.preprocessing import NLPPreprocessing


def main(args=None):
    print("Create Bag Of Words model and perform classification.")


def train_model(raw_corpus):
    # Pre-processing
    texts = NLPPreprocessing.preprocess_text(raw_corpus)
    processed_corpus = NLPPreprocessing.more_than_once(texts)

    # Create dictionary from corpus
    dictionary = corpora.Dictionary(processed_corpus)

    # Create Bag Of Words model
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    return bow_corpus, dictionary


def apply_model(dictionary, raw_corpus):
    # Pre-processing
    texts = NLPPreprocessing.preprocess_text(raw_corpus)
    processed_corpus = NLPPreprocessing.more_than_once(texts)

    # Apply model for new data
    bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]

    return bow_corpus


if __name__ == '__main__':
    # Show help message
    main()

    # Load train and test data
    train_labels, raw_train_corpus = load_as_df(NLPClassification.FILE_TRAIN_BANK)
    test_labels, raw_test_corpus = load_as_df(NLPClassification.FILE_TEST_BANK)

    # Train model using train data
    train_vectors, dictionary = train_model(raw_train_corpus)

    # Apply trained model
    test_vectors = apply_model(dictionary, raw_test_corpus)

    # Transform vectors for scikit
    train_vectors = matutils.corpus2csc(train_vectors)
    test_vectors = matutils.corpus2csc(test_vectors)

    classifier = NLPClassification()

    nb_classifier = classifier.train_nb_classifier(train_vectors.transpose(), train_labels)
    classifier.test_classifier(nb_classifier, test_vectors.transpose(), test_labels)

    nb_classifier = classifier.train_nb2_classifier(train_vectors.transpose().toarray(), train_labels)
    classifier.test_classifier(nb_classifier, test_vectors.transpose().toarray(), test_labels)

    classifier.test_all_classifiers(train_vectors.transpose().toarray(), train_labels,
                                    test_vectors.transpose().toarray(), test_labels)
