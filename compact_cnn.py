import tensorflow as tf
import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import pickle
import datetime
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
        self.base_dir = base_dir
        self.rst = rst
        self.lr = lr
        self.seg_his = {'loss': [], 'val_loss': []}
        self.seg_model = self.segment_model()
        self.cls_model = self.classification_model()

    def _conv_block(self, x, filters, kernel_size=3,
                    strides=1, activation='relu', name=None):
        if name:
            cname = 'conv-' + name
            bname = 'bn-' + name
        else:
            cname = bname =  None

        out = Conv2D(filters, kernel_size=kernel_size,
                    strides=strides, activation=activation,
                    kernel_initializer='random_normal',padding='same',
                    name=cname)(x)
        return BatchNormalization(name=bname)(out)


    def segment_model(self):
        image = Input(shape=(self.rst, self.rst, 1))

        # Block 1
        x = self._conv_block(image, 32, kernel_size=9, strides=2, name='seg-1')
        x = self._conv_block(x, 32, kernel_size=9, name='seg-2')
        x = self._conv_block(x, 32, kernel_size=9, name='seg-3')

        # Block 2
        x = self._conv_block(x, 64, kernel_size=7, strides=2, name='seg-4')
        x = self._conv_block(x, 64, kernel_size=7, name='seg-5')
        x = self._conv_block(x, 64, kernel_size=7, name='seg-6')

        # Block 3
        x = self._conv_block(x, 128, kernel_size=3, name='seg-7')
        x = self._conv_block(x, 128, kernel_size=3, name='seg-8')
        x = self._conv_block(x, 128, kernel_size=3, name='featmap')

        # Segmentation
        seg = self._conv_block(x, 1, kernel_size=1,
                            activation='tanh',name='segmap')

        model = Model(inputs=image, outputs=seg)
        model.compile(optimizer='adadelta', loss='mean_squared_error')
        return model


    def classification_model(self):
        seg_map_1 = GlobalMaxPooling2D()(self.seg_model.get_layer('bn-segmap').output)
        bn1 = BatchNormalization()(seg_map_1)

        seg_map_2 = GlobalAveragePooling2D()(self.seg_model.get_layer('bn-segmap').output)
        bn2 = BatchNormalization()(seg_map_2)

        bn3 = self._conv_block(self.seg_model.get_layer('bn-featmap').output,
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

        for i in range(len(model.layers)):
            name = model.layers[i].name
            if any([x in name for x in ['seg', 'featmap']]):
                model.layers[i].trainable = False

        model.compile(optimizer='adadelta', loss='binary_crossentropy')

        return model


    def compact_cnn_model(self):
        model = Model(
            inputs=self.cls_model.input,
            outputs=[
                self.cls_model.get_layer('bn-segmap').output,
                self.cls_model.get_layer('clsout').output,
            ]
        )
        model.compile(optimizer=Adam(self.lr),
                    loss=['mean_squared_error', 'binary_crossentropy'],
        )

        return model


    def train_model(self, model, data_gen, epochs, sample_weight):
        history = {"loss": [], "val_loss": []}
        if model == 'seg':
            train_model = self.seg_model
            next_batch = data_gen.next_seg_batch
        else:
            train_model = self.cls_model
            next_batch = data_gen.next_cls_batch

        for e in epochs:
            start_time = datetime.datetime.now()
            print("Train epochs {}/{} - ".format(e, epochs), end="")
            batch_loss = batch_val_loss = []
            for img, mask in next_batch():
                loss, val_loss = train_model.train_on_batch(img, mask)
                batch_loss.append(loss)
                batch_val_loss.append(loss)

            mean_batch_loss = np.mean(np.array(batch_loss))
            mean_batch_val_loss = np.mean(np.array(batch_val_loss))
            history['loss'].append(mean_batch_loss)
            history['val_loss'].append(mean_batch_val_loss)
            print("Loss: {}, Val Loss: {} - {}".format(
                mean_batch_loss, mean_batch_val_loss,
                datetime.datetime.now() - start_time
            ))

        return history

    def train(self, data_gen, seg_epochs, cls_epochs, sample_weight=None):
        self.seg_his = self.train_model('seg', data_gen, seg_epochs, sample_weight)
        self.cls_his = self.train_model('cls', data_gen, cls_epochs, sample_weight)
        self.plot_history()


    def plot_history(self):
        his = self.seg_his.history
        plt.plot(his['loss'], label='train loss')
        plt.plot(his['val_loss'], label='val loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('Segmentation model')
        plt.legend()
        plt.show()


    def show_output(self, x, y, idx):
        shape = (1, self.rst, self.rst, 1)
        image = x[idx].reshape((1, self.rst, self.rst, 1))
        mask = y[idx].reshape((1, self.rst//4, self.rst//4, 1))

        seg = self.seg_model.predict(image)
        score = self.cls_model.predict(image)
        plt.figure(figsize=(18,10))
        plt.subplot(131)
        plt.imshow(image[0].reshape(self.rst,self.rst))
        plt.subplot(132)
        segmap = seg[0].reshape((self.rst//4,self.rst//4))
        plt.imshow(segmap.astype('uint8'))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.2, 0.1, f'Score={score[0]}', fontsize=14,
                verticalalignment='top', bbox=props)
        plt.subplot(133)
        plt.imshow(mask[0].reshape(self.rst//4,self.rst//4))
        plt.show()

    def save_weight(self):
        self.seg_model.save_weights(self.base_dir + '/seg_model.h5')
        self.cls_model.save_weights(self.base_dir + '/cls_model.h5')

    def load_weight(self):
        self.seg_model.load_weights(self.base_dir + '/seg_model.h5')
        self.cls_model.load_weights(self.base_dir + '/cls_model.h5')

