#!/usr/bin/ python
# -*- coding: utf-8 -*-
# *************************************** #
#
#  Author:  Semen Budenkov
#  Date:    27/02/2017
#
# *************************************** #


import csv
import pandas as pd


def load_as_csv(file_name):
    with open(file_name) as opened_file:
        data = csv.reader(opened_file, delimiter='\t')

    return data


def load_as_df(file_name):
    print("Open file: {}".format(file_name))
    with open(file_name) as opened_file:
        data = pd.read_csv(opened_file, sep='\t', header=0)

    print("Number of sentences: {}".format(len(data['text'])))

    return data['sentiment'], data['text']


def main(args=None):
    print("This module provides functions for interacting with SentiRuEval set.")


if __name__ == '__main__':
    main()
