#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    27/02/2017
#
# *************************************** #


import logging
import langid
from collections import defaultdict
import scipy


logging.getLogger(__name__).addHandler(logging.NullHandler())


STOPLIST = set('for a of the and to in'.split(' '))

class NLPPreprocessing(object):
    def __init__(self, lng_list=None):
        if lng_list is None:
            langid.set_languages(['en', 'ru'])
        else:
            langid.set_languages(lng_list)

        self.logger = logging.getLogger(__name__)

    def detect_language(self, text):
        language = langid.classify(text)
        self.logger.info('Language: %s Confidence: %s',
                         language[0],
                         language[1])
        # Return only abbreviation
        return language[0]

    @staticmethod
    def preprocess_text(raw_corpus):
        # Split sentences by white space and filter out stopwords
        texts = [[word for word in document.lower().split() if word not in STOPLIST]
                 for document in raw_corpus]

        return texts

    @staticmethod
    def more_than_once(texts):
        # Count word frequencies
        frequency = defaultdict(int)
        for text in texts:
            for token in text:
                frequency[token] += 1

        # Only keep words that appear more than once
        return [[token for token in text if frequency[token] > 1] for text in texts]

    @staticmethod
    def cos_cdist(matrix, vector):
        """
        Compute the cosine distances between each row of matrix and vector.
        """
        v = vector.reshape(1, -1)
        return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)
