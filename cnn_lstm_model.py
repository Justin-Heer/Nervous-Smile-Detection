# import necessary libraries

# import python built-in libraries
import os
import cv2
import dlib
import re
from glob import glob

# import external libraries
import tensorflow as tf
from tensorflow_addons.metrics.f_scores import F1Score
import numpy as np
from imutils.face_utils import rect_to_bb

# import keras functions
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.layers import Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping

# import sklearn functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# import plotting utilities
import matplotlib.pyplot as plt
import seaborn as sns


# a function that creates a CNN model
def create_cnn():
    # create the pre-trained VGG-Face model
    cnn_model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), weights=None)

    # load the weights from the pre-trained model
    cnn_model.load_weights('rcmalli_vggface_tf_notop_resnet50.h5')

    # compile the model
    cnn_model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    # return the cnn model
    return cnn_model


# a function that creates a bidirectional LSTM model
def create_bi_lstm(n_frames, n_tokens):
    # create a LSTM model
    lstm_model = Sequential()

    # create a bidirectional LSTM-RNN
    lstm_model.add(Bidirectional(LSTM(units=64, return_sequences=False),
                                 input_shape=(n_frames, n_tokens)))

    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dropout(0.7))
    lstm_model.add(Dense(256, activation='relu'))
    lstm_model.add(Dropout(0.7))
    # two categories: happy/amused smiles and nervous/awkward smiles
    lstm_model.add(Dense(2, activation='softmax'))

    lstm_model.compile(loss='categorical_crossentropy', optimizer='adam',
                       metrics=['accuracy', F1Score(num_classes=2, average='micro')])

    # return the model
    return lstm_model


# a function that extracts features from a given frame
def extract_features_frame(frame, cnn_model):
    # get face detector from dlib
    detector = dlib.get_frontal_face_detector()

    # convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect a valid face from the frame
    faces = detector(frame_gray, 1)

    # return None if not exactly one face is detected
    if len(faces) != 1:
        return None

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
    try:
        face = cv2.resize(frame[x:x + w, y:y + h], (224, 224), interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(e)
        print(f'x={x},y={y},w={w},h={h}')
        print(f'The shape of the face is {frame[x:x + w, y:y + h].shape}')
        print(f'The shape of the frame is {frame_gray.shape}')
        # skip this frame and return None
        return None

    # expand the shape of the image by one dimension
    face_in = np.expand_dims(face, axis=0)

    # convert to dtype 'float64'
    face_in = face_in.astype('float64')

    # some preprocessing required by the model
    face_in = preprocess_input(face_in, version=2)

    # feature extraction using CNN
    feature = cnn_model.predict(face_in).ravel()

    # return the extracted feature
    return feature


# a function that extracts features from a given smile video
def extract_features(vpath, cnn_model):
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

            # extract features from the frame
            feature = extract_features_frame(frame, cnn_model)

            # add the extracted features to the list
            # if not None
            if feature is not None:
                features.append(feature)

    # if features is not empty
    if len(features) > 0:
        # convert to ndarray and return the results
        return np.array(features)
    else:
        # return None, otherwise
        return None


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
        if feature is not None:
            features.append(feature)

    # if features is not empty
    if len(features) > 0:
        # convert to ndarray and return the results
        return np.array(features)
    else:
        # return None, otherwise
        return None


# a function that extracts features from all the frames in a given folder
def extract_features_dir_2(fdir, cnn_model):
    # check the existence of the directory
    if not os.path.exists(fdir):
        raise ValueError('The directory "{}" does not exist!'.format(fdir))

    # create an empty list to hold the features
    # for all videos in the folder
    features = []

    # list all subdirectories
    sub_dirs = [sub_dir for sub_dir in os.listdir(fdir)
                if os.path.isdir(os.path.join(fdir, sub_dir))]

    # get the number part of the file name using the regular expression
    reg1 = re.compile('_(\d*)')

    # sort the file names
    sub_dirs.sort(key=lambda x: int(reg1.search(x).group(1)))

    for sub_dir in sub_dirs:

        # create an empty list to store features for each video
        features_video = []

        # form a path for each sub-folder
        sub_dir_path = os.path.join(fdir, sub_dir)

        # print the path we are currently processing
        print(sub_dir_path)

        # get the list of frames in this directory
        fpaths = glob(os.path.join(sub_dir_path, '*.jpg'))

        # get the number part of the file name using the regular expression
        reg2 = re.compile('\D*(\d*)[.]jpg')

        # sort the file names
        fpaths.sort(key=lambda x: int(reg2.search(x).group(1)))

        # extract features from each video
        for fpath in fpaths:

            # read in the image
            frame = cv2.imread(fpath)

            # extract features from the frame
            features_tmp = extract_features_frame(frame, cnn_model)

            # add the extracted features to the list
            # if not None
            if features_tmp is not None:
                features_video.append(features_tmp)

        # aggregate the extracted features of each video
        # to a single list if not empty
        if len(features_video) > 0:
            features.append(np.array(features_video))

    # if the resulting features is not empty
    if len(features) > 0:
        # convert to ndarray and return the results
        return np.array(features)
    else:
        # return None, otherwise
        return None


# a function that generates batches
# namely, zero-pad to make time-series
# of the same length
# this is a preprocessing step for BiLSTM
def generate_batch(X, y, batch_size, max_frames):
    # compute the number of batches
    n_batches = int(np.ceil(len(X) / batch_size))

    # a while loop to generate mini-batches
    # and return it after each call as a generator
    while True:

        #  process each batch by their batch id : Idx
        for idx in range(0, n_batches):

            # get the start position
            # of this batch in samples
            start = idx * batch_size

            # get the end position of this batch
            # in samples, note that the data at the
            # end position is NOT included in this batch
            end = np.min([(idx + 1) * batch_size, len(X)])

            # an empty list to hold the results
            # as a container
            data = []

            # put each sample into this batch
            # in a for loop
            for k in range(start, end):
                # get the kth sample
                # in the sample ndarray
                f = X[k]
                # get the length of this sample
                # i.e. the length of video frames
                frames = f.shape[0]
                # we shorten the video if it is longer
                # than a maximum length we set
                if frames > max_frames:
                    # crop the video to maximum length
                    f = f[0:max_frames, :]
                    # append the result to container list
                    data.append(f)
                # we pad zeros if otherwise
                elif frames < max_frames:
                    # create a temp sample
                    # of the maximum length
                    temp = np.zeros(shape=(max_frames, f.shape[1]))
                    # assign the shorter data sample
                    # to this longer data sample
                    temp[0:frames, :] = f
                    # append the result to container list
                    data.append(temp)
                else:
                    # no preprocess need
                    # if the data sample is
                    # of the same length as max_frames
                    data.append(f)
            # produce the result as a generator
            yield np.array(data), y[start:end]


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
        dir_happy = os.path.join('.', 'happy_frames')

        # construct the path for the directory of awkward smiles
        dir_awkward = os.path.join('.', 'nervous_frames')

        # create a VGG-Face CNN model
        cnn_model = create_cnn()

        # get the features for happy smiles
        features_happy = extract_features_dir_2(dir_happy, cnn_model)

        # get the features for awkward smiles
        features_awkward = extract_features_dir_2(dir_awkward, cnn_model)

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

    # a callback that interrupts training early when there is no more progress
    # to avoid wasting time and resources
    early_stopper = EarlyStopping(patience=10, verbose=1, restore_best_weights=True)

    # calculate the number of batches

    # for the training set
    n_batches_train = int(np.ceil(len(X_train) / batch_size))

    # for the validation set
    n_batches_val = int(np.ceil(len(X_val) / batch_size))

    # fit the model to data
    h = lstm_model.fit(train_gen,
                       epochs=N_EPOCHS,
                       steps_per_epoch=n_batches_train,
                       validation_data=val_gen,
                       validation_steps=n_batches_val,
                       verbose=1, callbacks=[opt_saver, early_stopper])

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
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.plot(h.history["accuracy"], label="accuracy")
    plt.plot(h.history["val_accuracy"], label="val_accuracy")
    plt.plot(h.history["f1_score"], label="f1_score")
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

    # load weights for the model
    if os.path.exists('biLSTM_tuned.h5'):
        print('load fine-tuned weights~!')
        lstm_model.load_weights('biLSTM_tuned.h5')
    else:
        print('load optimal weights from this training~!')
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

    # print the statistics

    # calculate the number of batches
    # for the test set
    n_batches_test = int(np.ceil(len(X_test) / batch_size))

    # get the prediction
    y_prediction = lstm_model.predict(test_gen,
                                      steps=n_batches_test,
                                      verbose=1)

    # get the predicted labels for each sample
    # y_pred = [np.argmax(y_prediction[i]) for i in range(y_prediction.shape[0])]
    y_pred = np.argmax(y_prediction, axis=-1)

    # get the ground truth labels
    y_true = [np.argmax(y_test[i]) for i in range(y_test.shape[0])]

    # use strings to represent each category
    target_names = ['nervous', 'happy']

    # compute the statistics
    print(classification_report(y_true, y_pred, target_names=target_names))

    # compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # print the confusion matrix
    print(cm)

    # plot the confusion matrix
    # as a seaborn heatmap

    # plot the heat map
    heat_map = sns.heatmap(cm, annot=True, cmap='hot')

    # set the tick labels
    heat_map.set_xticklabels(['Predicted Nervous', 'Predicted Happy'],
                             fontdict={'horizontalalignment': 'center'})

    heat_map.set_yticklabels(['True Nervous', 'True Happy'],
                             fontdict={'verticalalignment': 'center'})

    # set the title
    heat_map.set_title('The Confusion Matrix')

    # show the plot
    plt.show()


if __name__ == '__main__':
    main()
