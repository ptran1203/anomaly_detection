
import numpy as np
import utils
from collections import Counter
import os
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, base_dir, batch_size, prune=None):
        self.base_dir = base_dir
        self.batch_size = batch_size

        self.x = utils.pickle_load(os.path.join(self.base_dir, 'dataset/imgs.pkl'))
        self.y = utils.pickle_load(os.path.join(self.base_dir, 'dataset/labels.pkl'))
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


    def augment_array(self, x, y, labels, augment_factor):
        imgs = []
        masks = []
        labels = []
        for i in range(len(x)):
            imgs.append(x[i])
            labels.append(y[i])
            for _ in range(augment_factor):
                _x, _y = self.augment_one(x[i], y[i])
                imgs.append(_x)
                masks.append(_y)
                labels.append(labels[i])

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

