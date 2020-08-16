# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:09:25 2020

@author: justin
"""

import cv2
import os
import sys


basePath = 'GIFGIF_dataset\\embarrassment'
imageFiles = os.listdir(basePath)

imageFiles.sort(key=lambda x: int(x[:-4]))

validPath = 'GIFGIF_dataset\\valid'
invalidPath = 'GIFGIF_dataset\\invalid'

imagePaths = [os.path.join(basePath, imageFile) for imageFile in imageFiles]

for imageFile in imageFiles:

    imagePath = os.path.join(basePath, imageFile)

    cap = cv2.VideoCapture(imagePath)
    ret, frame = cap.read()
    print('a = valid, l = invalid')
    while(True):
        ret, frame = cap.read()
        if cv2.waitKey(34) & 0xFF == ord('q') or ret is False:
            valid = cv2.waitKey(-1)
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow('frame', frame)

    if valid == ord('a'):
        src = imagePath
        dst = os.path.join(validPath, imageFile)
        os.rename(src, dst)
    elif valid == ord('l'):
        src = imagePath
        dst = os.path.join(invalidPath, imageFile)
        os.rename(src, dst)
    elif valid == ord('t'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit()
