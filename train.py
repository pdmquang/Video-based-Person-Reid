from __future__ import division, print_function, absolute_import

import os
import pickle
import time
from keras.callbacks import ModelCheckpoint

# import utils.cuda_util
from random import shuffle

import numpy as np
import tensorflow as tf
from keras.applications import ResNet50

from tensorflow.python.keras import backend as K
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout, Conv2D, Reshape, Softmax
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adagrad
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from baseline.img_utils import get_random_eraser, crop_generator


def load_data(LIST, TRAIN):
    images, labels = [], []
    label_mapping = {}

    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        

        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            if last_label != lbl:
                label_cnt += 1
                label_mapping[label_cnt] = lbl
                
            last_label = lbl
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.resnet50.preprocess_input(img)

            images.append(img[0])
            labels.append(label_cnt)
            

    img_cnt = len(labels)
    shuffle_idxes = list(range(img_cnt))
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)

    return images, labels


def softmax_model_pretrain(train_list, train_dir, class_count, target_model_path):
    images, labels = load_data(train_list, train_dir)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.compat.v1.InteractiveSession(config=config)
    K.set_session(sess)

    # load pre-trained resnet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_count, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    net = Model(inputs=[base_model.input], outputs=[x])

    net.summary()

    for layer in net.layers:
        layer.trainable = True

    # pretrain
    batch_size = 16
    train_cnt = len(labels)
    train_datagen = ImageDataGenerator(
        shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2,
        horizontal_flip=0.5).flow(
        images[:train_cnt//10*9], labels[:train_cnt//10*9], batch_size=batch_size)

    val_datagen = ImageDataGenerator(horizontal_flip=0.5).flow(images[train_cnt//10*9:], labels[train_cnt//10*9:], batch_size=batch_size)


    save_best = ModelCheckpoint(target_model_path, monitor='val_accuracy', save_best_only=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{target_model_path} {str(time.time())}")

    net.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit(
        train_datagen, 
        epochs=40,
        validation_data=val_datagen,
        callbacks=[save_best, tensorboard_callback]
    )

    net.save(target_model_path)


def softmax_pretrain_on_dataset(source, project_path='.', dataset_parent='./Market-1501-v15.09.15'):
    if source == 'market':
        train_list = project_path + '/dataset/market_train.list'
        train_dir = dataset_parent + '/bounding_box_train'
        class_count = 751
    elif source == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        train_dir = dataset_parent + '/grid_label'
        class_count = 250
    elif source == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        train_dir = dataset_parent + '/cuhk01'
        class_count = 971
    elif source == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        train_dir = dataset_parent + '/viper'
        class_count = 630
    elif source == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        train_dir = dataset_parent + '/DukeMTMC-reID/train'
        class_count = 702
    elif 'grid-cv' in source:
        cv_idx = int(source.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        train_dir = dataset_parent + '/underground_reid/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in source:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    softmax_model_pretrain(train_list, train_dir, class_count, './reid_model/' + source + '_softmax_pretrain.h5')

if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    sources = ['market']
    for source in sources:
        softmax_pretrain_on_dataset(source, project_path='.', dataset_parent='./Market-1501-v15.09.15')
