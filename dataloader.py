
import numpy as np
import utils
from collections import Counter
import os
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, base_dir, batch_size, mode = 1, cls=1, prune=None, de_norm=False):
        TRAIN = 1
        TEST = 2

        self.base_dir = base_dir
        self.batch_size = batch_size
        ds_dir = os.path.join(self.base_dir, 'dataset/class_{}'.format(cls))
        if mode == TRAIN:
            self.x = utils.pickle_load(ds_dir + '/imgs_train.pkl')
            self.y = utils.pickle_load(ds_dir + '/marks_train.pkl')
        elif mode == TEST:
            self.x = utils.pickle_load(ds_dir + '/imgs_test.pkl')
            self.y = utils.pickle_load(ds_dir + '/marks_test.pkl')
        else:
            raise("Invalid option, should be one {} or {}".format(TRAIN, TEST))

        if de_norm:
            self.x = utils.de_norm(self.x)
            self.y = utils.de_norm(self.y)

        self.labels = np.array([1 if np.sum(mask) > 0 else 0 for mask in self.y])

        if prune is not None:
            self.x, self.y, self.labels = utils.prune(self.x, self.y, self.labels, prune)

        self.x = utils.norm(self.x)
        self.y = utils.norm(self.y)
        self.classes = np.unique(self.labels)
        self.per_class_ids = {}
        ids = np.array(range(len(self.x)))
        for c in self.classes:
            self.per_class_ids[c] = ids[self.labels == c]

        print(Counter(self.labels))


    def get_samples_for_class(self, c, samples=None):
        if samples is None:
            samples = self.batch_size
        try:
            np.random.shuffle(self.per_class_ids[c])
            to_return = self.per_class_ids[c][0:samples]
            return self.dataset_x[to_return]
        except:
            random = np.arange(self.dataset_x.shape[0])
            np.random.shuffle(random)
            to_return = random[:samples]
            return self.dataset_x[to_return]


    def augment_one(self, x, y):
        seed = np.random.randint(0, 100)
        new_x = utils.transform(x, seed)
        new_y = utils.transform(y, seed)
        return new_x, new_y


    def augment_array(self, x, y, z, augment_factor):
        imgs = []
        masks = []
        labels = []
        for i in range(len(x)):
            imgs.append(x[i])
            masks.append(y[i])
            labels.append(z[i])
            for _ in range(augment_factor):
                _x, _y = self.augment_one(x[i], y[i])
                imgs.append(_x)
                masks.append(_y)
                labels.append(z[i])

        return np.array(imgs), np.array(masks), np.array(labels)


    def next_batch(self, augment_factor):
        x = self.x
        y = self.y
        labels = self.labels

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, x.shape[0] - self.batch_size + 1, self.batch_size):
            access_pattern = indices[start_idx:start_idx + self.batch_size]

            yield self.augment_array(
                x[access_pattern, :, :, :],
                y[access_pattern],
                labels[access_pattern],
                augment_factor,
            )

