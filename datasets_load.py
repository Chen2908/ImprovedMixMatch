import os
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import itertools
import random
from sklearn.model_selection import ShuffleSplit


def load_cifar100():
    dataset = tfds.load(name='cifar100')
    train = dataset['train']
    test = dataset['test']
    data = train.concatenate(test)
    return data


def load_dataset(dataset_name):
    """
    The function loads the datasets from Tensorflow datasets.
    The datasets are returned as 'dataset' object, without a train-test split.
    :param dataset_name: the name of the dataset.
    'cifar100' for cifar100, 'cifar10' for cifar 10, 'svhn_cropped' for svhn, 'svhn+extra' for svhn extra or
    'stl10' for STL10
    :return:
    """
    dataset = tfds.load(name=dataset_name)
    train = dataset['train']
    test = dataset['test']
    if dataset_name == 'svhn+extra':
        train.concatenate(dataset['extra'])
    dataset = train.concatenate(test)

    return dataset


def split_cifar100_to_superclasses(data, superclasses_num=20):
    """
    Splits the CIFAR100 dataset according to the pre-defined superclasses.
    :param data: CIFAR100 as 'dataset' object
    :param superclasses_num: the amount of superclasses that will be returned.
    :return: a list with the required amount of superclasses
    """
    superclasses = {k: [] for k in range(superclasses_num)}
    for item in data:
        superclasses[int(item['coarse_label'])].append({'image': item['image'], 'label': item['label']})
    for cl in range(superclasses_num):
        superclasses[cl] = adjust_superclass_labels(superclasses[cl])
        superclasses[cl] = img_list_to_tf_dataset(superclasses[cl])
    return superclasses


def split_dataset_to_mini_datasets(dataset, num_of_splits, ds_dir, num_labels):
    """
    Splits the given dataset to smaller datasets in a stratified manner, and save them.
    :param dataset: the given dataset
    :param num_of_splits: required amount of smaller datasets
    :param ds_dir: where to save the new small datasets
    :param num_labels: how many labels each one of the small datasets will contain.
    :return: a list which contains the smaller datasets.
    """
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    dataset_divided_by_labels = {}
    # split the dataset to lists by the labels
    for item in dataset:
        if int(item['label']) not in dataset_divided_by_labels.keys():
            dataset_divided_by_labels[int(item['label'])] = []
        dataset_divided_by_labels[int(item['label'])].append({
            'image': item['image'],
            'label': item['label']
        })
    # split each one of the labels into even chunks
    labels_split_into_chunks = {}
    for label in range(num_labels):
        label_split_list = []
        label_split = np.array_split(np.array(dataset_divided_by_labels[label]), num_of_splits)
        for l in label_split:
            label_split_list.append(l.tolist())
        labels_split_into_chunks[label] = label_split_list
    mini_datasets_as_lists = []
    for i in range(len(label_split)):
        labels_list = [v[i] for v in labels_split_into_chunks.values()]
        labels_list = list(itertools.chain.from_iterable(labels_list))
        random.shuffle(labels_list)
        labels_list = img_list_to_tf_dataset(labels_list)
        export_tfrecord_dataset(f'{ds_dir}/partial_ds_{i}.tfrecord', labels_list)
        mini_datasets_as_lists.append(labels_list)
    return mini_datasets_as_lists


def split_labeled_unlabeled_overlap(train_ds, num_labeled, val_ds, num_classes):
    """
    split a given dataset to labeled dataset and unlabeled dataset
    :param train_ds: the dataset
    :param num_labeled: the size of the labeled set
    :param val_ds:
    :param num_classes: how many labeles the dataset contains
    :return: labeled and unlabeled sets
    """
    dataset = train_ds.shuffle(buffer_size=1000)
    counter = {}
    labeled = []
    unlabeled = []
    for example in iter(dataset):
        label = int(example['label'])
        if str(label) in counter.keys():
            counter[str(label)] += 1
        else:
            counter[str(label)] = 1
        if counter[str(label)] <= (num_labeled / num_classes):
            labeled.append(example)
            continue
        else:
            unlabeled.append({
                'image': example['image'],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
    for example in iter(val_ds):
        unlabeled.append({
            'image': example['image'],
            'label': tf.convert_to_tensor(-1, dtype=tf.int64)
        })
    random.shuffle(unlabeled)
    return labeled, unlabeled


def labeled_unlabeled_validation_split(train_index, val_index, train_val_ds):
    """
    split a given dataset to labeled dataset and unlabeled dataset
    :param train_index: the indices of the train set
    :param val_index: the indices of the validation set
    :param train_val_ds: the dataset
    :return: labeled, unlabeled and validation sets
    """
    train_records = []
    val_records = []
    unlabeled_records = []
    labeled_records = []
    index = 0
    for example in iter(train_val_ds):
        if index in train_index:
            train_records.append(example)
        elif index in val_index:
            val_records.append(example)
            unlabeled_records.append({
                'image': example['image'],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)})
        index += 1
    val_ds = list_to_tf_dataset_string(val_records)
    train_ds = list_to_tf_dataset_string(train_records)
    train_y = [int(example['label']) for example in iter(train_ds)]
    # train_size = the number of labelled examples
    splits = ShuffleSplit(n_splits=1, train_size=100,
                      random_state=42)
    for labeled_index, unlabeled_index in splits.split(train_y):
        index = 0
        for example in iter(train_ds):
            if index in labeled_index:
                labeled_records.append(example)
            else:
                unlabeled_records.append({
                    'image': example['image'],
                    'label': tf.convert_to_tensor(-1, dtype=tf.int64)})
            index += 1
    labeled_ds = list_to_tf_dataset_string(labeled_records)
    unlabeled_ds = list_to_tf_dataset_string(unlabeled_records)
    return labeled_ds, unlabeled_ds, val_ds


def img_list_to_tf_dataset(dataset):
    """
    convert images list to tensorflow dataset
    :param dataset: list of images
    :return: tensorflow dataset
    """
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image': tf.uint8, 'label': tf.int64},
        output_shapes={'image': (32, 32, 3), 'label': ()}
    )


def normalize_image(image, start=(0., 255.), end=(-1., 1.)):
    image = (image - start[0]) / (start[1] - start[0])
    image = image * (end[1] - end[0]) + end[0]
    return image


def process_parsed_dataset(dataset, num_classes, model_name=None):
    images = []
    labels = []
    for example in iter(dataset):
        decoded_image = tf.io.decode_png(example['image'], channels=3, dtype=tf.uint8)
        normalized_image = normalize_image(tf.cast(decoded_image, dtype=tf.float32))
        images.append(normalized_image)
        one_hot_label = tf.one_hot(example['label'], depth=num_classes, dtype=tf.float32)
        labels.append(one_hot_label)
    if model_name == 'baseline':
        return tf.data.Dataset.from_tensor_slices((images,labels))
    return tf.data.Dataset.from_tensor_slices({
        'image': images,
        'label': labels
    })


def list_to_tf_dataset_string(dataset):
    def _dataset_gen():
        for example in dataset:
            yield example
    return tf.data.Dataset.from_generator(
        _dataset_gen,
        output_types={'image': tf.string, 'label': tf.int64},
        output_shapes={'image': (), 'label': ()}
    )


def split_dataset(dataset, num_labelled, num_validations, num_classes):
    dataset = dataset.shuffle(buffer_size=10000)
    counter = [0 for _ in range(num_classes)]
    labelled = []
    unlabelled = []
    validation = []
    for example in iter(dataset):
        label = int(example['label'])
        counter[label] += 1
        if counter[label] <= (num_labelled / num_classes):
            labelled.append(example)
            continue
        elif counter[label] <= (num_validations / num_classes + num_labelled / num_classes):
            validation.append(example)
        unlabelled.append({
            'image': example['image'],
            'label': tf.convert_to_tensor(-1, dtype=tf.int64)
        })
    labelled = img_list_to_tf_dataset(labelled)
    unlabelled = img_list_to_tf_dataset(unlabelled)
    validation = img_list_to_tf_dataset(validation)
    return labelled, unlabelled, validation


def split_labeled_unlabeled_no_overlap(train_ds, num_labeled, num_classes):
    """
    split a given dataset to labeled dataset and unlabeled dataset
    :param train_ds: the dataset
    :param num_labeled: the size of the labeled set
    :param num_classes: how many labeles the dataset contains
    :return: labeled and unlabeled sets
    """
    dataset = train_ds.shuffle(buffer_size=100)
    counter = {}
    labeled = []
    unlabeled = []
    for example in iter(dataset):
        label = int(example['label'])
        if str(label) in counter.keys():
            counter[str(label)] += 1
        else:
            counter[str(label)] = 1
        if counter[str(label)] <= (num_labeled / num_classes):
            labeled.append(example)
            continue
        else:
            unlabeled.append({
                'image': example['image'],
                'label': tf.convert_to_tensor(-1, dtype=tf.int64)
            })
    return labeled, unlabeled


def export_tfrecord_dataset(dataset_path, dataset):
    """
    save the dataset as tfrecord file at the dataset_path
    :param dataset_path: where to dave the file
    :param dataset: the given dataset
    """
    serialized_dataset = dataset.map(tf_serialize_example)
    writer = tf.data.experimental.TFRecordWriter(dataset_path)
    writer.write(serialized_dataset)


def serialize_example(image, label):
    image = tf.image.encode_png(image)
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def tf_serialize_example(example):
    tf_string = tf.py_function(
        serialize_example,
        (example['image'], example['label']),
        tf.string
    )
    return tf.reshape(tf_string, ())


def load_tfrecord_dataset(dataset_path):
    raw_dataset = tf.data.TFRecordDataset([dataset_path])
    parsed_dataset = raw_dataset.map(_parse_function)
    return parsed_dataset


def _parse_function(example):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example, feature_description)


def adjust_superclass_labels(superclass):
    """
    adjust the labels of a given dataset to a range of (0-labels_num)
    :param superclass: the given dataset
    :return: the dataset with the fixed labels
    """
    labels_mapping = {}
    new_labeled_superclass = []
    for example in iter(superclass):
        label = int(example['label'])
        if str(label) not in labels_mapping.keys():
            new_label = len(labels_mapping)
            labels_mapping[str(label)] = new_label
        label = labels_mapping[str(label)]
        new_labeled_superclass.append({
            'image': example['image'],
            'label': tf.convert_to_tensor(label, dtype=tf.int64)
        })
    return new_labeled_superclass


def divide_cifar100_superclass_to_subdataset(ds_dir, num_of_splits):
    """
    divide the CIFAR100 superclass to smaller sets and save each set as a tfrecord file.
    :param ds_dir: where to save the smaller datasets
    :param num_of_splits: how manu sets to create
    """
    if not os.path.exists(ds_dir):
        os.mkdir(ds_dir)
    data = load_cifar100()
    superclasses = split_cifar100_to_superclasses(data, superclasses_num=20)
    for i in superclasses.keys():
        superclass_folder = f'{ds_dir}/cifar100_superclass_{i}'
        if not os.path.exists(superclass_folder):
            os.mkdir(superclass_folder)
        new_labeled_superclass = adjust_superclass_labels(superclasses[i])
        split_dataset_to_mini_datasets(new_labeled_superclass, num_of_splits, superclass_folder)


def get_cifar_10(batch_num):
    data = load_dataset('cifar10')
    data_as_list = [x for x in iter(data)]
    assert (batch_num <= 1), "CIFAR10 Dataset don't have this subset, try 0 or 1"
    labels = [i for i in range(0, 10)]
    random.shuffle(labels)
    labels = np.reshape(labels, (2, 5))
    num_each_label = int(len(data_as_list) / 5)
    first = True
    all = None
    for label in labels[batch_num]:
        ds_l = data.filter(lambda d: d.get("label") == label).take(num_each_label)
        if first:
            all = ds_l
            first = False
        else:
            all = all.concatenate(ds_l)
    y_ds = tf.cast(list(labels[batch_num]), tf.int64)
    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(y_ds),
            values=tf.constant(list(range(len(y_ds))))
        ),
        default_value=tf.constant(-1),
        name="class"
    )
    all = all.map(lambda d: (d.get('image'), table.lookup(d.get('lable'))))
    all = all.shuffle(len(data_as_list), shuffle_each_iteration=True)
    return all, 5
