# import necessary libraries
import os
import cv2
import dlib
import sys
import re
import numpy as np
from glob import glob
import tensorflow as tf
from imutils.face_utils import rect_to_bb
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.layers import Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# a function that creates a CNN model
def create_cnn():
    # create the pre-trained VGG-Face model
    cnn_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3))

    # load the weights from the pre-trained model
    # cnn_model.load_weights('rcmalli_vggface_tf_notop_resnet50.h5')

    # compile the model
    cnn_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    # return the cnn model
    return cnn_model


# a function that creates a bidirectional LSTM model
def create_bi_lstm(n_frames, n_tokens):
    # create a LSTM model
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(units=256, return_sequences=False),
                                 input_shape=(n_frames, n_tokens)))
    lstm_model.add(Dense(512, activation='relu'))
    lstm_model.add(Dropout(0.5))
    # two categories: happy/amused smiles and nervous/awkward smiles
    lstm_model.add(Dense(2, activation='softmax'))
    lstm_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # return the model
    return lstm_model


# a function that extracts features from a given smile video
def extract_features(vpath, cnn_model):
    # get face detector from dlib and initialize the face aligner
    detector = dlib.get_frontal_face_detector()

    # open the video file
    video_cap = cv2.VideoCapture(vpath)

    # create a flag variable to control the loop
    success = True

    # an empty list to hold the results
    features = []

    # process each video frame
    while success:

        # read a frame from the video
        success, frame = video_cap.read()

        # process this frame if successful
        if success:
            # convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect a valid face from the frame
            faces = detector(frame_gray, 2)

            # continue if more than one face is detected
            if len(faces) != 1:
                continue

            # get the coordinates of the bounding box
            (x, y, w, h) = rect_to_bb(faces[0])

            # crop the negative values
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0

            # resize the image to 224x224
            # face = imutils.resize(frame[x:x + w, y:y + h], width=224, height=224)
            try:
                face = cv2.resize(frame[x:x + w, y:y + h], (224, 224), interpolation=cv2.INTER_AREA)
            except Exception as e:
                print(e)
                print(f'x={x},y={y},w={w},h={h}')
                print(f'The shape of the face is {frame[x:x + w, y:y + h].shape}')
                print(f'The shape of the frame is {frame_gray.shape}')
                print(f'The file is {vpath}')
                sys.exit(-1)

            # expand the shape of the image by one dimension
            face_in = np.expand_dims(face, axis=0)

            # convert to dtype 'float64'
            face_in = face_in.astype('float64')

            # some preprocessing required by the model
            face_in = preprocess_input(face_in, version=2)

            # feature extraction using CNN
            feature = cnn_model.predict(face_in).ravel()

            # add to the list
            features.append(feature)

    # convert to ndarray and return the results
    return np.array(features)


# a function that extracts features from all videos in a given folder
def extract_features_dir(vdir, cnn_model):
    # check the existence of the directory
    if not os.path.exists(vdir):
        raise ValueError('The directory "{}" does not exist!'.format(vdir))

    # create an empty list to hold the result
    features = []

    # get the list of mp4 video files in this
    vpaths = glob(os.path.join(vdir, '*.mp4'))

    # get the number part of the file name using regular expressions
    r = re.compile('\D*(\d*).mp4')

    # sort the file names
    vpaths.sort(key=lambda x: int(r.search(x).group(1)))

    # extract features from each video
    for vpath in vpaths:
        print(f'Current File:{vpath}')
        feature = extract_features(vpath, cnn_model)
        if feature.size != 0:
            features.append(feature)

    # return the results as a ndarray
    return np.array(features)


# a function that generates batches
# namely, zero-pad to make time-series
# of the same length
def generate_batch(x_samples, y_samples, batch_size, expected_frames):
    num_batches = len(x_samples) // batch_size

    while True:
        for batchIdx in range(0, num_batches):
            start = batchIdx * batch_size
            end = (batchIdx + 1) * batch_size
            x_data = []
            for k in range(start, end):
                x = x_samples[k]
                frames = x.shape[0]
                if frames > expected_frames:
                    x = x[0:expected_frames, :]
                    x_data.append(x)
                elif frames < expected_frames:
                    temp = np.zeros(shape=(expected_frames, x.shape[1]))
                    temp[0:frames, :] = x
                    x_data.append(temp)
                else:
                    x_data.append(x)

            yield np.array(x_data), y_samples[start:end]


# the start point of this script
def main():
    # get available GPUs
    gpu = tf.config.experimental.list_physical_devices('GPU')

    # get available CPUs
    cpu = tf.config.experimental.list_physical_devices('CPU')

    # set the memory growth to True
    tf.config.experimental.set_memory_growth(gpu[0], True)

    # check if saved files exist
    if not os.path.exists('dataset.npz'):
        # construct the path for the directory of happy smiles
        dir_happy = os.path.join('dataset-video', 'Happy')

        # construct the path for the directory of awkward smiles
        dir_awkward = os.path.join('dataset-video', 'Awkward')

        # create a VGG-Face CNN model
        cnn_model = create_cnn()

        # get the features for happy smiles
        features_happy = extract_features_dir(dir_happy, cnn_model)

        # get the features for awkward smiles
        features_awkward = extract_features_dir(dir_awkward, cnn_model)

        # combine these features
        features = np.concatenate((features_happy, features_awkward), axis=0)

        # generate corresponding labels
        labels = np.concatenate((np.ones(len(features_happy)), np.zeros(len(features_awkward))), axis=0)

        # save features and labels to file
        np.savez_compressed('dataset', f=features, l=labels)

    else:
        # load features from the saved file
        data = np.load('dataset.npz')

        # turn on allow_pickle
        data.allow_pickle = True

        # fetch the data
        features = data.get('f')

        # fetch the data
        labels = data.get('l')

    # split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                        random_state=16)

    # split the dataset into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=16)

    # calculate the max length of any video in the dataset
    expected_frames = max([X_train[i].shape[0] for i in range(len(X_train))])

    # set the batch size
    batch_size = 16

    # Converts a class vector (integers) to binary class matrix
    # for use with categorical_crossentropy
    # IMPORTANT for CATEGORICAL_CROSSENTROPY!
    y_train = np_utils.to_categorical(y_train, 2)
    y_val = np_utils.to_categorical(y_val, 2)

    # generator for training the LSTM model
    train_gen = generate_batch(X_train, y_train, batch_size, expected_frames)

    # generator for validating the LSTM model
    val_gen = generate_batch(X_val, y_val, batch_size, expected_frames)

    # create the bidirectional LSTM model
    lstm_model = create_bi_lstm(expected_frames, X_train[0].shape[1])

    # a callback function that saves the optimal model during training
    opt_saver = ModelCheckpoint('biLSTM.h5', save_best_only=True,
                                monitor='val_accuracy',
                                verbose=1,
                                mode='max')

    # set the epochs
    N_EPOCHS = 30

    # fit the model to data
    h = lstm_model.fit(train_gen,
                       epochs=N_EPOCHS,
                       steps_per_epoch=len(X_train) // batch_size,
                       validation_data=val_gen,
                       validation_steps=len(X_val) // batch_size,
                       verbose=1, callbacks=[opt_saver])

    # retrieve the optimal results
    val_accu_max = max(h.history["val_accuracy"])
    val_accu_max_ind = np.argmax(h.history["val_accuracy"])

    # print the training results
    print('Training Results:')
    print(f'The maximum validation accuracy is {val_accu_max * 100:0.2f}%')
    print(f'Epoch # is {val_accu_max_ind + 1}')

    # plot the training + validation loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N_EPOCHS), h.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N_EPOCHS), h.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N_EPOCHS), h.history["accuracy"], label="accuracy")
    plt.plot(np.arange(0, N_EPOCHS), h.history["val_accuracy"], label="val_accuracy")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

    # if 'q' key is pressed, stop the loop
    # print('Press q to exit and any other key to continue~')
    # key = cv2.waitKey(0)
    # check the pressed key
    # if (key & 0xFF) == ord("q"):
    #    sys.exit(0)

    # the start of the testing part
    print('The testing of the trained model~')

    # load the weights for the optimal model
    lstm_model.load_weights('biLSTM.h5')

    # Converts a class vector (integers) to binary class matrix
    # for use with categorical_crossentropy
    # IMPORTANT for CATEGORICAL_CROSSENTROPY!
    y_test = np_utils.to_categorical(y_test, 2)

    # generator for validating the LSTM model
    test_gen = generate_batch(X_test, y_test, batch_size, expected_frames)

    # checks the models performance
    accu_test = lstm_model.evaluate(test_gen,
                                    steps=len(X_test) // batch_size,
                                    verbose=1)

    # print the test result
    print(f'The test accuracy for this model is {accu_test[1] * 100:0.2f}%')


if __name__ == '__main__':
    main()
