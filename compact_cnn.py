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

class CompactModel:
    def __init__(self, rst, lr, base_dir):
        self.rst = rst
        self.lr = lr
        self.base_dir = base_dir
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
                    loss=['mean_squared_error', 'binary_crossentropy'],
                    # loss_weights=[1, 0.7]
        )

        return model

    def train(self, x, y, label, seg_epochs, cls_epochs, sample_weight=None):
        print("Train segmetation model")
        self.seg_his = self.seg_model.fit(x, y,
                                          epochs=seg_epochs,
                                          batch_size=64,
                                          validation_split=0.1,
                                          sample_weight=sample_weight)
        print("Train classification model")
        self.cls_his = self.compact_model.fit(x, [y, label],
                                          epochs=cls_epochs,
                                          batch_size=64,
                                          validation_split=0.1,
                                          sample_weight=[
                                            sample_weight,
                                            sample_weight
                                        ])
        
    def show_output(self, x, y, idx):
        mask_rst = self.rst//8
        shape = (1, self.rst, self.rst, 1)
        image = x[idx].reshape((1, self.rst, self.rst, 1))
        mask = y[idx].reshape((1, mask_rst, mask_rst, 1))

        seg, score = self.compact_model.predict(image)
        plt.figure(figsize=(18,10))
        plt.subplot(131)
        plt.imshow(image[0].reshape(self.rst,self.rst))
        plt.subplot(132)
        segmap = seg[0].reshape((28,28))
        plt.imshow(segmap.astype('uint8'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.2, 0.1, f'Score={score[0]}', fontsize=14,
                verticalalignment='top', bbox=props)
        plt.subplot(133)
        plt.imshow(mask[0].reshape(mask_rst, mask_rst))
        plt.show()

    def save_weight(self):
        self.seg_model.save_weights(self.base_dir + '/seg_model.h5')
        self.cls_model.save_weights(self.base_dir + '/cls_model.h5')

    def load_weight(self):
        self.seg_model.load_weights(self.base_dir + '/seg_model.h5')
        self.cls_model.load_weights(self.base_dir + '/cls_model.h5')

