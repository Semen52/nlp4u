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
            print(classification_report(test_labels, prediction_nb))
            print("--------------------------------------------------------------------------------")

    def test_all_classifiers(self,
                             train_vectors, train_labels,
                             test_vectors, test_labels):
        names = ["Nearest Neighbors",
                 "Linear SVM",
                 "RBF SVM",
                 # "Gaussian Process",
                 "Decision Tree",
                 "Random Forest",
                 "Neural Net",
                 "AdaBoost",
                 "Naive Bayes",
                 "QDA"]

        classifiers = [KNeighborsClassifier(3),
                       SVC(kernel="linear", C=0.025),
                       SVC(gamma=2, C=1),
                       # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
                       DecisionTreeClassifier(max_depth=5),
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

    FILE_TRAIN_BANK = '../../data/external/sentirueval-2015/mystem/bank_train_mystem.tsv'
    FILE_TEST_BANK = '../../data/external/sentirueval-2015/mystem/bank_test_etalon_mystem.tsv'


def main(args=None):
    print("Classification module.")


if __name__ == '__main__':
    # Show help message
    main()
