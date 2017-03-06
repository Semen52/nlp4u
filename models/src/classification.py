#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    27/02/2017
#
# *************************************** #


import time
import logging

import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


logging.getLogger(__name__).addHandler(logging.NullHandler())


class NLPClassification(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def train_nb_classifier(self, train_vectors, train_labels):
        # Perform classification with naïve Bayes classifier
        classifier_nb = MultinomialNB()
        t0 = time.time()
        classifier_nb.fit(train_vectors, train_labels)
        t1 = time.time()
        time_nb_train = t1 - t0

        self.logger.info("Multinomial Naive Bayes classifier. Training time: ",
                         time_nb_train)

        return classifier_nb

    def train_nb2_classifier(self, train_vectors, train_labels):
        # Perform classification with naïve Bayes classifier
        classifier_nb = GaussianNB()
        t0 = time.time()
        classifier_nb.fit(train_vectors, train_labels)
        t1 = time.time()
        time_nb_train = t1 - t0

        self.logger.info("Gaussian Naive Bayes classifier. Training time: ",
                         time_nb_train)

        return classifier_nb

    def test_classifier(self, classifier, test_vectors, test_labels=None):
        t0 = time.time()
        prediction_nb = classifier.predict(test_vectors)
        score = classifier.score(test_vectors, test_labels)
        t1 = time.time()
        time_nb_predict = t1 - t0

        self.logger.info("Naive Bayes classifier. Prediction time: ",
                         time_nb_predict)

        if test_labels is None:
            return  prediction_nb
        else:
            print("\n")
            print("--------------------------------------------------------------------------------")
            print("Naive Bayes classifier")
            print("--------------------------------------------------------------------------------")
            print(score)
            print("--------------------------------------------------------------------------------")
            print(classification_report(test_labels, prediction_nb))
            print("--------------------------------------------------------------------------------")
            self.calculate_accuracy(test_labels, prediction_nb)
            print("--------------------------------------------------------------------------------")

    def test_all_classifiers(self,
                             train_vectors, train_labels,
                             test_vectors, test_labels):
        names = ["Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 "Log Reg",
                 # "Gaussian Process",
                 # "Decision Tree",
                 "Random Forest",
                 "Neural Net",
                 "AdaBoost",
                 "Naive Bayes",
                 "QDA"]

        classifiers = [KNeighborsClassifier(3),
                       SVC(kernel="linear", C=0.025),
                       SVC(gamma=2, C=1),
                       linear_model.LogisticRegression(C=1e5),
                       # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                       # DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                       MLPClassifier(alpha=1),
                       AdaBoostClassifier(),
                       GaussianNB(),
                       QuadraticDiscriminantAnalysis()]

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            clf.fit(train_vectors, train_labels)
            score = clf.score(test_vectors, test_labels)
            prediction = clf.predict(test_vectors)

            print("\n")
            print("--------------------------------------------------------------------------------")
            print(name)
            print("--------------------------------------------------------------------------------")
            print(score)
            print("--------------------------------------------------------------------------------")
            print(classification_report(test_labels, prediction))
            print("--------------------------------------------------------------------------------")
            self.calculate_accuracy(test_labels, prediction)
            print("--------------------------------------------------------------------------------")

    def calculate_accuracy(self,
                           true_labels,
                           predicted_labels):
        pos_tp = 0
        pos_tn = 0
        pos_fp = 0
        pos_fn = 0

        neg_tp = 0
        neg_tn = 0
        neg_fp = 0
        neg_fn = 0

        true_labels = np.asarray(true_labels)
        predicted_labels = np.asarray(predicted_labels)

        if true_labels.shape != predicted_labels.shape:
            print("ERROR: shapes are not equals")
            return

        for ix, resultValue in enumerate(predicted_labels):
            etalonValue = true_labels[ix]

            if etalonValue == resultValue:
                if etalonValue == 'positive':
                    pos_tp += 1
                elif etalonValue == 'negative':
                    neg_tp += 1

            if etalonValue != resultValue:
                if etalonValue != 'positive':
                    if resultValue == 'positive':
                        pos_fp += 1

                if etalonValue != 'negative':
                    if resultValue == 'negative':
                        neg_fp += 1

                if etalonValue == 'positive':
                    pos_fn += 1
                elif etalonValue == 'negative':
                    neg_fn += 1

            if etalonValue != 'positive':
                if resultValue != 'positive':
                    pos_tn += 1

            if etalonValue != 'negative':
                if resultValue != 'negative':
                    neg_tn += 1

        # print('Counts    - ', 'positive:', pos_tp, pos_tn, pos_fp, pos_fn,
        #                       'negative:', neg_tp, neg_tn, neg_fp, neg_fn)

        precision_pos = 0
        precision_neg = 0
        recall_pos = 0
        recall_neg = 0

        if (pos_tp + pos_fp) > 0:
            precision_pos = pos_tp / (pos_tp + pos_fp)
        if (neg_tp + neg_fp) > 0:
            precision_neg = neg_tp / (neg_tp + neg_fp)

        if (pos_tp + pos_fn) > 0:
            recall_pos = pos_tp / (pos_tp + pos_fn)
        if (neg_tp + neg_fn) > 0:
            recall_neg = neg_tp / (neg_tp + neg_fn)
        F_pos = 0
        F_neg = 0

        if (precision_pos + recall_pos) > 0:
            F_pos = 2 * (
                (precision_pos * recall_pos) / ((precision_pos + recall_pos)))
        if (precision_neg + recall_neg) > 0:
            F_neg = 2 * (
                (precision_neg * recall_neg) / ((precision_neg + recall_neg)))

        F_R = (F_pos + F_neg) / 2

        print('Prec_neg - ', precision_neg, '\tRec_neg - ', recall_neg, '\tF_neg - ', F_neg)
        print('Prec_pos - ', precision_pos, '\tRec_pos - ', recall_pos, '\tF_pos - ', F_pos)
        print('F_R      - ', F_R)


    FILE_TRAIN_BANK = '../../data/external/sentirueval-2015/mystem/bank_train_mystem.tsv'
    FILE_TEST_BANK = '../../data/external/sentirueval-2015/mystem/bank_test_etalon_mystem.tsv'
    FILE_TRAIN_TTK = '../../data/external/sentirueval-2015/mystem/ttk_train_mystem.tsv'
    FILE_TEST_TTK = '../../data/external/sentirueval-2015/mystem/ttk_test_etalon_mystem_debug.tsv'


def main(args=None):
    print("Classification module.")


if __name__ == '__main__':
    # Show help message
    main()
