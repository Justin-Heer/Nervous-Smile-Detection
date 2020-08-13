# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 07:03:34 2020

@author: justin

Annotation optimal human level accuracy score test.

The purpose of this script is to watch each video and classify it as either
happy or nervous, we will use the accuracy score generated as a benchmark for
our algorithm
"""

import os
import cv2
import random
import sys
import csv


def create_image_paths(path):
    # looks inside path dir and returns a list of paths to
    # each item in the dir

    return [os.path.join(path, imagePath) for imagePath in os.listdir(path)]


if os.path.exists('annotation_imageDirs.csv'):

    with open('annotation_imageDirs.csv', newline='') as file:
        reader = csv.reader(file)
        imageDirs = list(reader)[0]

    with open('pred_y.csv', newline='') as file:
        reader = csv.reader(file)
        y_pred = list(reader)[0]

    with open('test_y.csv', newline='') as file:
        reader = csv.reader(file)
        y = list(reader)[0]

else:
    # Create base path for dirs
    happyPath = r'dataset/happy_frames/'
    nervPath = r'dataset/nervous_frames'

    # Get image paths and store in variable
    happyDirs = create_image_paths(happyPath)
    nervDirs = create_image_paths(nervPath)

    # Initial shuffle of variables starting with a random seed
    random.seed(random.randint(1, 10))
    random.shuffle(happyDirs)
    random.shuffle(nervDirs)

    # Create array of labels
    y_happy = [0] * len(happyDirs)
    y_nerv = [1] * len(nervDirs)

    # Join lists together and then shuffle again 
    imageDirs = happyDirs + nervDirs
    y = y_happy + y_nerv

    temp = list(zip(imageDirs, y))

    random.shuffle(temp)

    # Separate labels and dirs
    imageDirs, y = zip(*temp)
    y_pred = []

for imageDir in imageDirs[len(y_pred):]:
    print('a = happy, l = nervous, t = replay, y = quit')
    while True:
        try:
            # Get images from image dir
            images = os.listdir(imageDir)

            # Sort images
            images.sort(key=lambda x: int(x[0:-4]))

            # Start looping through images
            for image in images:
                imagePath = os.path.join(imageDir, image)
                array = cv2.imread(imagePath)
                cv2.imshow('current', array)
                cv2.waitKey(5)

            # Collect input
            key = chr(cv2.waitKey(-1))
            if key == 'a':
                print('you predicted happy')
                y_pred.append(0)
            elif key == 'l':
                print('you predicted nervous')
                y_pred.append(1)
            elif key == 't':
                raise ValueError
            elif key == 'y':
                # Write relevant files to csv
                with open('annotation_imageDirs.csv', 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(imageDirs)

                with open('pred_y.csv', 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(y_pred)

                with open('test_y.csv', 'w', newline='') as myfile:
                    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    wr.writerow(y)

                cv2.destroyAllWindows()
                sys.exit()

            cv2.destroyAllWindows()
            break
        except ValueError:
            print('Replaying video')
            pass

print('all images processed')
# Write y_pred to csv
with open('pred_y.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(y_pred)
