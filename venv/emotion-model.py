import re

import pandas as pd
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report

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


def make_model(img_height, img_width):
    model = Sequential([
        # Input(shape=(img_height, img_width, 1)),
        InputLayer(input_shape=(img_height, img_width, 1)),
        Conv2D(36, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=3, strides=2),
        Conv2D(64, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=3, strides=2),
        Conv2D(128, kernel_size=3, activation='relu'),
        MaxPool2D(pool_size=3, strides=2),
        Flatten(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dense(7, activation='softmax', name='race', kernel_regularizer=l1(1))
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model


def get_data(data_dir, img_height, img_width):
    img_data_arr = []
    class_names = []
    for label in os.listdir(data_dir):
        last_person = ''
        for file in os.listdir(os.path.join(data_dir, label)):
            # skip images on the same person
            curr_person = re.search('^(S[0-9]+)_', file)
            if curr_person == last_person:
                continue
            last_person = curr_person

            try:
                image_path = os.path.join(data_dir, label, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (img_height, img_width))  # Reshaping images to preferred size
                image = np.array(image)
                img_data_arr.append(image)
                class_names.append(label)
            except Exception as e:
                print(e)
    return np.array(img_data_arr), np.array(class_names)


def main():
    # Parameters
    debug = False
    seed = np.random.seed(42)
    img_size = 48
    data_dir = 'assets/dataset/CK+48'
    # labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']
    target_encoder = OrdinalEncoder()

    # Load data
    # train_data = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     image_size=(img_size, img_size),
    #     color_mode='grayscale',
    #     seed=42,
    #     validation_split=0.2,
    #     subset='training'
    # )
    # test_data = tf.keras.preprocessing.image_dataset_from_directory(
    #     data_dir,
    #     image_size=(img_size, img_size),
    #     color_mode='grayscale',
    #     seed=42,
    #     validation_split=0.2,
    #     subset='validation'
    # )
    X, y = get_data(data_dir, img_size, img_size)
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(X[600])
        plt.title(y[600])
        plt.show()
    X = X.reshape(-1, img_size, img_size, 1)  # reshape for model
    y = target_encoder.fit_transform(y.reshape(y.shape[0], 1))  # encode as integers

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    dummy_y_train = np_utils.to_categorical(y_train)
    dummy_y_test = np_utils.to_categorical(y_test)

    # Normalise data
    X_train = X_train / 255
    X_test = X_test / 255

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

    # train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    # test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    model = make_model(img_size, img_size)
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=200, verbose=1,
                                   restore_best_weights=True)
    history = model.fit(datagen.flow(X_train, dummy_y_train, batch_size=32), validation_data=(X_test, dummy_y_test),
                        steps_per_epoch=len(X_train) / 32, epochs=3000, callbacks=[early_stopping])
    # history = model.fit(X_train, dummy_y_train, validation_data=(X_test, dummy_y_test), epochs=1500,
    #                     callbacks=[early_stopping])
    # history = model.fit(train_data, validation_data=test_data, epochs=500, callbacks=[early_stopping])

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
    plt.show()

    predictions = model.predict_classes(X_test)
    print(classification_report(y_test.ravel(), predictions, target_names=target_encoder.categories_[0]))

    model.save('models/my-emotion-model.hdf5')


if __name__ == '__main__':
    main()
