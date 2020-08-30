import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import pickle

from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import (
    Input, Dense, Reshape,
    Flatten, Dropout,
    BatchNormalization, Activation,
    Lambda,Layer, Add, Concatenate,
    Average,UpSampling2D,
    MaxPooling2D, AveragePooling2D,
    GlobalMaxPooling2D,GlobalAveragePooling2D,
)
from keras.models import Sequential, Model, model_from_json
from keras.optimizers import Adam


class AutoEncoder:
    def __init__(self, rst, lr, base_dir):
        self.rst = rst
        self.lr = lr
        self.base_dir = base_dir
        self.ae_model = self.autoencoder()

    def _conv_block(self, x, filters, kernel_size=3,
                    strides=1, activation='relu', name=None):
        out = Conv2D(filters, kernel_size=kernel_size,
                    strides=strides, activation=activation,
                    kernel_initializer='random_normal',padding='same')(x)
        return BatchNormalization(name=name)(out) \
                if name is not None \
                else BatchNormalization()(out)


    def _deconv_block(self, x, filters, kernel_size=3,
                    strides=1, activation='relu', name=None,
                    skip_con=None):
        out = Conv2DTranspose(filters, kernel_size=kernel_size,
                    strides=strides, activation=activation,
                    kernel_initializer='random_normal',padding='same')(x)
        if skip_con is not None:
            out = Concatenate()([out, skip_con])
        return BatchNormalization()(out)


    def autoencoder(self):
        image = Input(shape=(self.rst, self.rst, 1))

        # Encode
        x1 = self._conv_block(image, 1, 11, strides=1)
        x2 = self._conv_block(x1, 48, 9, strides=2)
        x3 = self._conv_block(x2, 48, 7, strides=2)
        x4 = self._conv_block(x3, 48, 5, strides=2)
        x5 = self._conv_block(x4, 48, 3, strides=2)

        # Decode
        up1 = self._deconv_block(x4, 48, 5, strides=2)
        up2 = self._deconv_block(up1, 48, 7, strides=2, skip_con=x4)
        up3 = self._deconv_block(up2, 48, 9, strides=2, skip_con=x3)
        up4 = self._deconv_block(up3, 48, 11, strides=2, skip_con=x2)

        out = self._conv_block(up4, 96, 1, 1, activation='tanh')

        model = Model(inputs=image, outputs=out)
        model.compile(optimizers=Adam(lr=self.lr), loss='mean_squared_error')

        return model


    def train(self, x, y, epochs, sample_weight=None):
        print("Train autoencoder model")
        self.ae_his = self.ae_model.fit(x, y,
                                        epochs=epochs,
                                        batch_size=64,
                                        validation_split=0.1,
                                        sample_weight=sample_weight)


    def show_output(self, x, y, idx):
        mask_rst = self.rst
        shape = (1, self.rst, self.rst, 1)
        image = x[idx].reshape((1, self.rst, self.rst, 1))
        mask = y[idx].reshape((1, mask_rst, mask_rst, 1))

        seg = self.ae_model.predict(image)
        plt.figure(figsize=(18,10))
        plt.subplot(131)
        plt.imshow(image[0].reshape(self.rst,self.rst))
        plt.subplot(132)
        segmap = seg[0].reshape((28,28))
        plt.imshow(segmap.astype('uint8'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.subplot(133)
        plt.imshow(mask[0].reshape(mask_rst, mask_rst))
        plt.show()


    def save_weight(self):
        self.ae_model.save_weights(self.base_dir + '/ae_model.h5')

    def load_weight(self):
        self.ae_model.load_weights(self.base_dir + '/ae_model.h5')
