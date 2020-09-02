import pickle
import numpy as np
import keras.preprocessing.image as image_processing

def pickle_save(object, path):
    try:
        print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except:
        print('save data to {} failed'.format(path))


def pickle_load(path):
    try:
        print("Loading data from {}".format(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
            print('load data successfully'.format(path))
            return data
    except Exception as e:
        print(str(e))
        return None


def prune(x, y, labels, prune_classes):
    """
    prune data by give classes
    """
    for class_to_prune in range(len(prune_classes)):
        remove_size = prune_classes[class_to_prune]
        if remove_size <= 0:
            continue
        print(class_to_prune)
        all_ids = list(np.arange(len(x)))
        mask = [lc == class_to_prune for lc in labels]
        all_ids_c = np.array(all_ids)[mask]
        np.random.shuffle(all_ids_c)
        to_delete  = all_ids_c[:remove_size]
        x = np.delete(x, to_delete, axis=0)
        y = np.delete(y, to_delete, axis=0)
        labels = np.delete(labels, to_delete, axis=0)
        print('Remove {} items in class {}'.format(remove_size, class_to_prune))
    return x, y, labels


def norm(imgs):
    return (imgs - 127.5) / 127.5

def de_norm(imgs):
    return imgs * 127.5 + 127.5


def transform(x, seed=0):
    np.random.seed(seed)
    img = image_processing.random_rotation(x, 0.2)
    img = image_processing.random_shear(img, 30)
    img = image_processing.random_zoom(img, (0.5, 1.1))
    if np.random.rand() >= 0.5:
        img = np.fliplr(img)

    return img


def weighted_samples(labels, weight_map):
    w = []
    for i in range(len(labels)):
        w.append(weight_map[labels[i]])
    
    return np.array(w)
