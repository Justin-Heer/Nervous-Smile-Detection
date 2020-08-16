# -*- coding: utf-8 -*-
"""
The purpose of this script is to test the output of 
'dataset-human-accuracy-score-generator.py' to determine the classification
accuracy of the tester. This script is intended for the researchers to use
so there are no failsafes or error messages implemented as its pretty simple

"""

import os
import csv
from sklearn.metrics import confusion_matrix, classification_report

if os.path.exists('pred_y.csv'):

    with open('pred_y.csv', newline='') as file:
        reader = csv.reader(file)
        y_pred = list(reader)[0]

    with open('test_y.csv', newline='') as file:
        reader = csv.reader(file)
        y = list(reader)[0]

print(classification_report(y, y_pred))

print(confusion_matrix(y, y_pred))
