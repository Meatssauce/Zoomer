import re

import pandas as pd
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report

from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Input, InputLayer, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D
from imblearn.over_sampling import RandomOverSampler


def make_model(img_height, img_width, img_depth):
    net = Sequential(name='DCNN')

    net.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(img_width, img_height, img_depth), activation='elu',
                   padding='same', kernel_initializer='he_normal', name='conv2d_1'))
    net.add(BatchNormalization(name='batchnorm_1'))
    net.add(Conv2D(filters=64, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer='he_normal',
                   name='conv2d_2'))
    net.add(BatchNormalization(name='batchnorm_2'))

    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
    net.add(Dropout(0.4, name='dropout_1'))

    net.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
                   name='conv2d_3'))
    net.add(BatchNormalization(name='batchnorm_3'))
    net.add(Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
                   name='conv2d_4'))
    net.add(BatchNormalization(name='batchnorm_4'))

    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
    net.add(Dropout(0.4, name='dropout_2'))

    net.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
                   name='conv2d_5'))
    net.add(BatchNormalization(name='batchnorm_5'))
    net.add(Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same', kernel_initializer='he_normal',
                   name='conv2d_6'))
    net.add(BatchNormalization(name='batchnorm_6'))

    net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
    net.add(Dropout(0.5, name='dropout_3'))

    net.add(Flatten(name='flatten'))
    net.add(Dense(128, activation='elu', kernel_initializer='he_normal', name='dense_1'))
    net.add(BatchNormalization(name='batchnorm_7'))
    net.add(Dropout(0.6, name='dropout_4'))
    net.add(Dense(7, activation='softmax', name='out_layer'))

    net.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    net.summary()

    return net


def get_data(data_dir, img_height, img_width):
    img_data_arr = []
    class_names = []
    df = pd.read_csv(data_dir)
    df['pixels'] = df['pixels'].str.split(' ')
    for i in df.index:
        pixels = [float(v) for v in df.loc[i, 'pixels']]
        img_data_arr.append(np.array(pixels).reshape(img_height, img_width))
        class_names.append(df.loc[i, 'emotion'])
    return np.array(img_data_arr), np.array(class_names)


def main():
    # Parameters
    debug = False
    seed = np.random.seed(42)
    img_size = 48
    data_dir = 'assets/dataset/fer2013/fer2013.csv'
    emotion_label_to_text = {
        0: 'anger',
        1: 'disgust',
        2: 'fear',
        3: 'happiness',
        4: 'sadness',
        5: 'surprise',
        6: 'neutral'
    }

    # Load data
    X, y = get_data(data_dir, img_size, img_size)
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(X[1112])
        plt.title(y[1112])
        plt.show()
    X = X.reshape(-1, img_size, img_size, 1)  # reshape for model
    y = y.reshape(y.shape[0], 1)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    dummy_y_train = np_utils.to_categorical(y_train)
    dummy_y_test = np_utils.to_categorical(y_test)

    # Normalise data
    # X_train = X_train / 255
    # X_test = X_test / 255

    # Compute class weights
    class_weight = utils.class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=np.ravel(y_train))
    class_weight = dict(enumerate(class_weight))

    # Augment data
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.2,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)

    model = make_model(img_size, img_size, 1)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=200, verbose=1,
                                   restore_best_weights=True)
    history = model.fit(datagen.flow(X_train, dummy_y_train, batch_size=32), validation_data=(X_test, dummy_y_test),
                        steps_per_epoch=len(X_train) / 32, epochs=3000, callbacks=[early_stopping],
                        class_weight=class_weight)
    # history = model.fit(X_train, dummy_y_train, validation_data=(X_test, dummy_y_test), epochs=1500,
    #                     callbacks=[early_stopping])

    model.save('models/my-emotion-model-4.hdf5')

    # Evaluate model
    acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training')
    plt.plot(val_acc, label='Validation')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training')
    plt.plot(val_loss, label='Validation')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('Figure_3.png')

    predictions = model.predict_classes(X_test)
    report = classification_report(y_test.ravel(), predictions,
                                   target_names=[v for k, v in emotion_label_to_text.items()])
    df = pd.DataFrame(report).transpose()
    df.to_csv('classification_report.csv')
    print(report)


if __name__ == '__main__':
    main()
