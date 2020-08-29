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

BASE_DIR = "/content/drive/My Drive/DAGN2007"



class CompactModel:
    def __init__(self, rst, lr):
        self.rst = rst
        self.lr = lr
        self.seg_model = self.segment_model()
        self.cls_model = self.classification_model()
        self.compact_model = self.compact_cnn_model()


    def _conv_block(self, x, filters, kernel_size=3,
                    strides=1, activation='relu', name=None):
        out = Conv2D(filters, kernel_size=kernel_size,
                    strides=strides, activation=activation,
                    kernel_initializer='random_normal',padding='same')(x)
        return BatchNormalization(name=name)(out) \
                if name is not None \
                else BatchNormalization()(out)


    def segment_model(self):
        image = Input(shape=(self.rst, self.rst, 1))

        # Block 1
        x = self._conv_block(image, 32, kernel_size=11, strides=2)
        x = self._conv_block(x, 32, kernel_size=11)
        x = self._conv_block(x, 32, kernel_size=11)

        # Block 2
        x = self._conv_block(x, 64, kernel_size=7, strides=2)
        x = self._conv_block(x, 64, kernel_size=7)
        x = self._conv_block(x, 64, kernel_size=7)

        # Block 3
        x = self._conv_block(x, 128, kernel_size=3, strides=2)
        x = self._conv_block(x, 128, kernel_size=3)
        x = self._conv_block(x, 128, kernel_size=3, name='featmap')

        # Segmentation
        seg = self._conv_block(x, 1, kernel_size=1, strides=1,
                            activation='tanh',name='segmap')

        model = Model(inputs=image, outputs=seg)
        model.compile(optimizer=Adam(self.lr), loss='mean_squared_error')
        return model


    def classification_model(self):
        seg_map_1 = GlobalMaxPooling2D()(self.seg_model.get_layer('segmap').output)
        bn1 = BatchNormalization()(seg_map_1)

        seg_map_2 = GlobalAveragePooling2D()(self.seg_model.get_layer('segmap').output)
        bn2 = BatchNormalization()(seg_map_2)

        bn3 = self._conv_block(self.seg_model.get_layer('featmap').output,
                                filters=32, kernel_size=1, strides=1)

        feat1 = GlobalMaxPooling2D()(bn3)
        bn4 = BatchNormalization()(feat1)

        feat2 = GlobalAveragePooling2D()(bn3)
        bn5 = BatchNormalization()(feat2)

        merged = Concatenate()([bn1, bn2, bn4, bn5])

        out = Dense(1, activation='sigmoid',
                    kernel_initializer='random_normal',
                    name='clsout')(merged)

        model = Model(inputs=self.seg_model.input, outputs=out)
        model.compile(optimizer=Adam(self.lr), loss='binary_crossentropy')

        return model


    def compact_cnn_model(self):
        model = Model(
            inputs=self.cls_model.input,
            outputs=[
                self.cls_model.get_layer('segmap').output,
                self.cls_model.get_layer('clsout').output,
            ]
        )
        model.compile(optimizer=Adam(self.lr),
                    loss=['mean_squared_error', 'binary_crossentropy'])

        return model
