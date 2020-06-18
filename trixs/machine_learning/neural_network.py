"""
Copyright 2018-2020 Toyota Resarch Institute. All rights reserved.
Use of this source code is governed by an Apache 2.0
license that can be found in the LICENSE file.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv1D, \
  MaxPooling1D, Flatten, Conv2D, MaxPooling2D

def classification_model(x_train, y_train, x_val, y_val, params):

    dropout = params['dropout']
    act = params['activation_function']
    optimizer = params['optimizer']

    layers = params['layers']

    model = Sequential()
    if x_val is None or y_val is None:
        validation_data = None
    else:
        x_val = np.expand_dims(x_val, axis=-1)
        validation_data = [x_val, y_val]


    if params['cnn']:
        x_train = np.expand_dims(x_train, axis=-1)
        model.add(Conv1D(filters=params['n_filters'],
                         kernel_size=(params['kernel']),
                         strides=(params['strides']), padding='valid',
                         activation=act,
                         input_shape=(x_train.shape[1], 1),
                         data_format="channels_last"))
        model.add(MaxPooling1D(pool_size=(params['pool_size']),
                               strides=None, padding='valid'))
        model.add(Flatten())
        model.add(Dense(layers[0], activation=act))
    else:
        model.add(Dense(layers[0], activation=act,
                    input_shape=(x_train.shape[1],)))
    model.add(Dropout(dropout))


    # note the change here relative to the regresion problem
    for ii in range(1, len(layers)):
        model.add(Dense(layers[ii], activation=act))
        model.add(Dropout(dropout))

    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss=params['loss_function'],
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                      validation_data=validation_data,
                      verbose=params.get('verbose',1),
                      shuffle=True)

    return history, model