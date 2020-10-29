# a script that computes the duration

# import the necessary libraries
import os
import cv2
import argparse
import subprocess
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True,
                        help='The directory that contains all the images of this sample')
    parser.add_argument('-f', '--file', required=True,
                        help='The video file that contains all the images of this sample')
    parser.add_argument('-t', '--threshold', required=False, default=5000000,
                        help='The threshold below which we should assume two images are identical')
    parser.add_argument('-l', '--link', required=True,
                        help='Please input the link for this video')
    parser.add_argument('-r', '--label', required=False, default='pos',
                        help='Please input the label for this video: "pos" or "neg"')

    args = vars(parser.parse_args())

    if not os.path.exists(args['dir']):
        print('The directory for the training sample doesn\'t exit!')
        SystemExit(-1)

    if not os.path.exists(args['file']):
        print('The video file doesn\'t exist!')
        SystemExit(-1)

    if args['label'] not in ['pos', 'neg']:
        print('Wrong label for this video!')
        SystemExit(-1)

    # open the csv file
    df = pd.read_csv('Smile_dataset_dataset_companion.csv')

    # create a filter
    filt = (df['filename'] == os.path.basename(args['dir'])) & \
           (df['pos/neg'] == args['label'])

    # the prefix of Youtube links
    prefix = 'https://www.youtube.com/watch?v='

    # update the link field
    if args['link'][: 5] == 'https':
        df.loc[filt, 'link'] = args['link']
    else:
        df.loc[filt, 'link'] = prefix + args['link']

    # get the file list
    lst = os.listdir(args['dir'])

    # sort the file name
    lst.sort(key=lambda x: int(x[:-4]))

    # construct the path of the start frame
    pathS = os.path.join(args['dir'], lst[0])

    # construct the path of the end frame
    # pathE = os.path.join(args['dir'], lst[len(lst)-1])

    # read in the start frame
    frameS = cv2.imread(pathS)

    # read in the end frame
    # frameE = cv2.imread(pathE)

    # get the frame rate of the video
    fps = subprocess.getoutput('ffprobe -v 0 -of compact=p=0 -select_streams 0 \
                                     -show_entries stream=r_frame_rate "{}"'.format(args['file']))
    fps = fps.split('=')[1]

    # convert the frame rate to integer
    fps = eval(fps)

    # open the video
    camera = cv2.VideoCapture(args['file'])

    # keep track of the frame number
    nframe = 0

    # record the start frame#
    sframe_num = -1

    # set the start minimum difference to the threshold
    diff_min = args['threshold']

    # keep looping
    while True:

        # grab the current frame
        (grabbed, frame) = camera.read()

        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if not grabbed:
            break

        # resize the frame to 480*360
        frame = cv2.resize(frame, (frameS.shape[1], frameS.shape[0]))

        # compute the difference between each frame
        diff = cv2.absdiff(frame, frameS)

        # count non-zero items
        diff_n = np.count_nonzero(diff)

        # compute the Structural Similarity Index (SSIM)
        # ssim_val = compare_ssim(frame, frameS, multichannel=True)

        # compare to the threshold
        if diff_n < diff_min:
            diff_min = diff_n
            sframe_num = nframe
            frames = frame

        # increase the frame counter
        nframe += 1

    if sframe_num == -1:
        print('Not found!')
    else:
        # start time
        minute_s = (sframe_num / fps) // 60
        second_s = (sframe_num / fps) - int(minute_s) * 60
        millisecond_s = (second_s - int(second_s)) * 1000

        print('Start time: {:02d}:{:02d}.{:03d}'.format(int(minute_s), int(second_s), int(millisecond_s)))
        # end time
        minute_e = ((sframe_num + len(lst) - 1) / fps) // 60
        second_e = ((sframe_num + len(lst) - 1) / fps) - int(minute_e) * 60
        millisecond_e = (second_e - int(second_e)) * 1000

        print('End time: {:02d}:{:02d}.{:03d}'.format(int(minute_e), int(second_e), int(millisecond_e)))

        # update start time
        df.loc[filt, 'start'] = '{:02d}:{:02d}.{:03d}'.format(int(minute_s), int(second_s), int(millisecond_s))

        # update end time
        df.loc[filt, 'end'] = '{:02d}:{:02d}.{:03d}'.format(int(minute_e), int(second_e), int(millisecond_e))

        # compute the Structural Similarity Index (SSIM)
        ssim_val = ssim(frames, frameS, multichannel=True)

        # Print the SSIM value
        print("The similarity index is {:.2f}%".format(ssim_val * 100))

        # save the file
        df.to_csv('Smile_dataset_dataset_companion.csv', index=False)
