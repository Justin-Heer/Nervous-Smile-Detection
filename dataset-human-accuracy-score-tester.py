# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 09:51:37 2020

@author: justi
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
