# import built-in libraries
import datetime

import numpy as np
import os

# import tuning utilities
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Bidirectional
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.backend import clear_session
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow_addons.metrics import F1Score

from cnn_lstm_model import generate_batch

# get available GPUs
gpu = tf.config.experimental.list_physical_devices('GPU')

# set the memory growth to True
tf.config.experimental.set_memory_growth(gpu[0], True)

# set the log directory to your previous one
# if you want to continue a previously stopped tuning
logdir = None

# set the root directory of the log
root = 'logs'

if logdir is None:
    # set the directory of the logs for this run
    # by formatting date and time
    # if you start a new tuning
    logdir = os.path.join(root, datetime.datetime.now().strftime('%Y%m%d-%H%M'))

# hyper parameters to experiment with
hp_model = hp.HParam('model', hp.Discrete(['resnet50']))
hp_units_lstm = hp.HParam('units_lstm', hp.Discrete(list(range(64, 320, 64))))
hp_units_layer1 = hp.HParam('units_layer1', hp.Discrete(list(range(64, 320, 64))))
hp_units_layer2 = hp.HParam('units_layer2', hp.Discrete(list(range(64, 320, 64))))
hp_rate_dropout = hp.HParam('rate_dropout', hp.Discrete([0.3, 0.5, 0.7]))
hp_optimizer = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'rmsprop']))
hp_batch_size = hp.HParam('batch_size', hp.Discrete(list(range(8, 20, 4))))

# set the metrics of training
metrics = ['accuracy', 'f1_score_nervous', 'f1_score_happy']

# write a top-level experiment configuration
with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(
        hparams=[hp_model, hp_units_lstm, hp_units_layer1, hp_units_layer2,
                 hp_rate_dropout, hp_optimizer, hp_batch_size],
        metrics=[hp.Metric(metrics[i], display_name=metrics[i]) for i in range(len(metrics))],
    )


# create a function that evaluates the model
def create_evaluate_model(hparams, run_dir):
    # load the dataset according the model

    # load features from the saved file
    data = np.load('dataset_{}.npz'.format(hparams[hp_model]))

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
    batch_size = hparams[hp_batch_size]

    # Converts a class vector (integers) to binary class matrix
    # for use with categorical_crossentropy
    # IMPORTANT for CATEGORICAL_CROSSENTROPY!
    y_train = np_utils.to_categorical(y_train, 2)
    y_val = np_utils.to_categorical(y_val, 2)

    # generator for training the LSTM model
    train_gen = generate_batch(X_train, y_train, batch_size, expected_frames)

    # generator for validating the LSTM model
    val_gen = generate_batch(X_val, y_val, batch_size, expected_frames)

    # create a lstm model
    lstm_model = Sequential([
        Bidirectional(LSTM(units=hparams[hp_units_lstm], return_sequences=False),
                      input_shape=(expected_frames, X_train[0].shape[1])),
        Dense(units=hparams[hp_units_layer1], activation='relu'),
        Dropout(rate=hparams[hp_rate_dropout]),
        Dense(units=hparams[hp_units_layer2], activation='relu'),
        Dropout(rate=hparams[hp_rate_dropout]),
        Dense(units=2, activation='softmax')]
    )

    # compile the model
    lstm_model.compile(loss='categorical_crossentropy', optimizer=hparams[hp_optimizer],
                       metrics=['accuracy', F1Score(num_classes=2)])

    # a callback that interrupts training early when there is no more progress
    # to avoid wasting time and resources
    early_stopper = EarlyStopping(patience=5, verbose=1, restore_best_weights=True)

    # calculate the number of batches

    # for the training set
    n_batches_train = int(np.ceil(len(X_train) / batch_size))

    # for the validation set
    n_batches_val = int(np.ceil(len(X_val) / batch_size))

    # train the model
    try:
        lstm_model.fit(train_gen,
                       epochs=30,
                       steps_per_epoch=n_batches_train,
                       validation_data=val_gen,
                       validation_steps=n_batches_val,
                       verbose=1, callbacks=[early_stopper,
                                             tf.keras.callbacks.TensorBoard(run_dir),
                                             hp.KerasCallback(run_dir, hparams)
                                             ])
    except Exception as e:
        print(e)
        print('Exception Occurred~Skip this result~!')
        return None

    # evaluate the model
    accu = lstm_model.evaluate(val_gen,
                               steps=n_batches_val,
                               verbose=1)

    # save the model

    # assign a path
    wpath = os.path.join(run_dir, 'model_weights_{}.h5'.format(os.path.basename(run_dir)))

    # save the model
    lstm_model.save_weights(wpath, save_format='h5')

    return accu


# create an index to denote the serial number of each run
n_run = 0

# start the tuning process
for model in hp_model.domain.values:
    for units_lstm in hp_units_lstm.domain.values:
        for units_layer1 in hp_units_layer1.domain.values:
            for units_layer2 in hp_units_layer2.domain.values:
                for rate_dropout in hp_rate_dropout.domain.values:
                    for optimizer in hp_optimizer.domain.values:
                        for batch_size in hp_batch_size.domain.values:
                            hparams = {
                                hp_model: model,
                                hp_units_lstm: units_lstm,
                                hp_units_layer1: units_layer1,
                                hp_units_layer2: units_layer2,
                                hp_rate_dropout: rate_dropout,
                                hp_optimizer: optimizer,
                                hp_batch_size: batch_size
                            }
                            print(f'starting trial # {n_run} ...')
                            print({h.name: hparams[h] for h in hparams})

                            # CAUTION: must create a new folder
                            # for each run
                            # set the folder name for this run
                            run_dir = os.path.join(logdir, 'run-{}-{}-{}-{}-{}-{}-{}'.format(
                                model, units_lstm, units_layer1, units_layer2,
                                rate_dropout, optimizer, batch_size
                            ))

                            # skip if this directory already exists
                            if os.path.exists(run_dir):
                                continue

                            # write down the results
                            with tf.summary.create_file_writer(run_dir).as_default():
                                # record the values used in this trial
                                hp.hparams(hparams, trial_id=f"{n_run:04d}")

                                # create and evaluate the model
                                accuracy = create_evaluate_model(hparams, run_dir)

                                # check the results
                                if accuracy is not None:
                                    print(f'The accuracy for this run is {100 * accuracy[1]:0.2f}')
                                    print(f'F1 score for nervous is {100 * accuracy[2][0]:0.2f}')
                                    print(f'F1 score for happy is {100 * accuracy[2][1]:0.2f}')
                                    tf.summary.scalar(metrics[0], accuracy[1], step=n_run)
                                    tf.summary.scalar(metrics[1], accuracy[2][0], step=n_run)
                                    tf.summary.scalar(metrics[2], accuracy[2][1], step=n_run)

                            # release the memory for this run
                            clear_session()

                            print(f'ending trial # {n_run} ...')
                            n_run += 1
