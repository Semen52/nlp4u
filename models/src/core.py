#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    30/01/2017
#
# *************************************** #


import logging


logging.getLogger(__name__).addHandler(logging.NullHandler())


class NLPCore(object):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def sentiment_analysis(self, text):
        return 'neutral'

    def classification(self, text):
        return ''

    def key_word_search(self, text):
        return ''
