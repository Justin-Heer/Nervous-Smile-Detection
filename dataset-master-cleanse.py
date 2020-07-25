# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 19:10:30 2020

@author: justin

Applying haar cascade classifier to dataset-master to filter down our dataset
"""
# Imports
import cv2
import os


# Functions
def detect_smile(image):

    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                          "haarcascade_smile.xml")
    smiles = smile_cascade.detectMultiScale(image, scaleFactor=1.1,
                                            minNeighbors=20)
    return smiles

# Script


folder_dir = 'dataset-master/'
image_files = os.listdir(folder_dir)
image_paths = [os.path.join(folder_dir, image_file) for
               image_file in image_files]
images = [cv2.imread(image_path) for image_path in image_paths]
smiles = [detect_smile(image) for image in images]

# %%
diff = len(smiles) - smiles.count(())

print(diff)
