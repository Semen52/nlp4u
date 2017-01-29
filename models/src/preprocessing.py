#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    30/01/2017
#
# *************************************** #


import logging
import langid


logging.getLogger(__name__).addHandler(logging.NullHandler())


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
