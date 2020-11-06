import os
import sys
sys.path.append(os.getcwd())

from vision.utils.tf_datagen import gen_fence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
import glob
import cv2
from IPython.display import clear_output
AUTOTUNE = tf.data.experimental.AUTOTUNE

import segmentation_models as sm

img_size = 128
n_channels = 3
batch_size = 1

DATA_PATH = '/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/train_set'
datagenerator = gen_fence(DATA_PATH, '/images/training/', '/images/validation/',batch_size, img_size, n_channels)
dataset = datagenerator.get_datasets()

N_CLASSES = datagenerator.n_classes

# -- Keras Functional API -- #
# -- UNet Implementation -- #
# Everything here is from tensorflow.keras.layers
# I imported tensorflow.keras.layers * to make it easier to read
# dropout_rate = 0.5
# input_size = (img_size, img_size, n_channels)

# If you want to know more about why we are using `he_normal`: 
# https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849#319849  
# Or the excelent fastai course: 
# https://github.com/fastai/course-v3/blob/master/nbs/dl2/02b_initializing.ipynb
# initializer = 'he_normal'


# # -- Encoder -- #
# # Block encoder 1
# inputs = Input(shape=input_size)
# conv_enc_1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=initializer)(inputs)
# conv_enc_1 = Conv2D(64, 3, activation = 'relu', padding='same', kernel_initializer=initializer)(conv_enc_1)

# # Block encoder 2
# max_pool_enc_2 = MaxPooling2D(pool_size=(2, 2))(conv_enc_1)
# conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_2)
# conv_enc_2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_2)

# # Block  encoder 3
# max_pool_enc_3 = MaxPooling2D(pool_size=(2, 2))(conv_enc_2)
# conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_3)
# conv_enc_3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_3)

# # Block  encoder 4
# max_pool_enc_4 = MaxPooling2D(pool_size=(2, 2))(conv_enc_3)
# conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(max_pool_enc_4)
# conv_enc_4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_enc_4)
# # -- Encoder -- #

# # ----------- #
# maxpool = MaxPooling2D(pool_size=(2, 2))(conv_enc_4)
# conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(maxpool)
# conv = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv)
# # ----------- #

# # -- Dencoder -- #
# # Block decoder 1
# up_dec_1 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv))
# merge_dec_1 = concatenate([conv_enc_4, up_dec_1], axis = 3)
# conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_1)
# conv_dec_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_1)

# # Block decoder 2
# up_dec_2 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_1))
# merge_dec_2 = concatenate([conv_enc_3, up_dec_2], axis = 3)
# conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_2)
# conv_dec_2 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_2)

# # Block decoder 3
# up_dec_3 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_2))
# merge_dec_3 = concatenate([conv_enc_2, up_dec_3], axis = 3)
# conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_3)
# conv_dec_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_3)

# # Block decoder 4
# up_dec_4 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = initializer)(UpSampling2D(size = (2,2))(conv_dec_3))
# merge_dec_4 = concatenate([conv_enc_1, up_dec_4], axis = 3)
# conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(merge_dec_4)
# conv_dec_4 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# conv_dec_4 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = initializer)(conv_dec_4)
# # -- Dencoder -- #

# output = Conv2D(N_CLASSES, 1, activation = 'softmax')(conv_dec_4)

# model = tf.keras.Model(inputs = inputs, outputs = output)
# model.compile(optimizer=Adam(learning_rate=0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])

def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
    """
    plt.figure(figsize=(18, 18))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    

# def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
#     """Return a filter mask with the top 1 predicitons
#     only.

#     Parameters
#     ----------
#     pred_mask : tf.Tensor
#         A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
#         N_CLASS values (vector) which represents the probability of the pixel
#         being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
#         has been predicted class 2 with a probability of 100%.

#     Returns
#     -------
#     tf.Tensor
#         A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
#         for each pixels.
#     """
#     # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
#     # 1 prediction for each class but we want the highest score only
#     # so we use argmax
#     pred_mask = tf.argmax(pred_mask, axis=-1)
#     # pred_mask becomes [IMG_SIZE, IMG_SIZE]
#     # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
#     pred_mask = tf.expand_dims(pred_mask, axis=-1)
#     return pred_mask
    
# def show_predictions(dataset=None, num=1):
#     """Show a sample prediction.

#     Parameters
#     ----------
#     dataset : [type], optional
#         [Input dataset, by default None
#     num : int, optional
#         Number of sample to show, by default 1
#     """
#     if dataset:
#         for image, mask in dataset.take(num):
#             pred_mask = model.predict(image)
#             display_sample([image[0], true_mask, create_mask(pred_mask)])
#     # else:
#     #     # The model is expecting a tensor of the size
#     #     # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
#     #     # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
#     #     # and we want only 1 inference to be faster
#     #     # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
#     #     one_img_batch = sample_image[0][tf.newaxis, ...]
#     #     # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
#     #     inference = model.predict(one_img_batch)
#     #     # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
#     #     pred_mask = create_mask(inference)
#     #     # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
#     #     display_sample([sample_image[0], sample_mask[0],pred_mask[0]])

# for image, mask in dataset['train'].take(1):
#     sample_image, sample_mask = image, mask

# show_predictions()

# class DisplayCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         clear_output(wait=True)
#         show_predictions()
#         print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

# EPOCHS = 20

# logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
# tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

# callbacks = [
#     # to show samples after each epoch
#     DisplayCallback(),
#     # to collect some useful metrics and visualize them in tensorboard
#     tensorboard_callback,
#     # if no accuracy improvements we can stop the training directly
#     tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
#     # to save checkpoints
#     tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
# ]

# model = tf.keras.Model(inputs = inputs, outputs = output)

# # # here I'm using a new optimizer: https://arxiv.org/abs/1908.03265
# optimizer=tfa.optimizers.RectifiedAdam(lr=1e-3)

# loss = tf.keras.losses.SparseCategoricalCrossentropy()

# model.compile(optimizer=optimizer, loss = loss,
#                   metrics=['accuracy'])

# STEPS_PER_EPOCH = datagenerator.train_data_size // batch_size
# VALIDATION_STEPS = datagenerator.val_data_size // batch_size

# model_history = model.fit(dataset['train'], epochs=EPOCHS,
#                     steps_per_epoch=STEPS_PER_EPOCH,
#                     validation_steps=VALIDATION_STEPS,
#                     validation_data=dataset['val'],
#                     callbacks=callbacks)

callbacks = [
    # to show samples after each epoch
    #DisplayCallback(),
    # to collect some useful metrics and visualize them in tensorboard
   # tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    #tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    #tf.keras.callbacks.ModelCheckpoint('best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

# define model
BACKBONE = 'resnet34'
EPOCHS = 10
STEPS_PER_EPOCH = datagenerator.train_data_size // batch_size
VALIDATION_STEPS = datagenerator.val_data_size // batch_size

preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet('resnet34', classes=1, activation='sigmoid', input_shape=(img_size, img_size, n_channels))
model.compile(
    'Adam',
    loss=sm.losses.binary_focal_loss,
    metrics=[sm.metrics.iou_score],
)
# model.load_weights('best_model_unet.h5')

model_history = model.fit(dataset['train'], epochs=EPOCHS,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_steps=VALIDATION_STEPS,
                    validation_data=dataset['val'],
                    callbacks=callbacks)

def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, IMG_SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
    
def show_predictions(dataset=None, num=1):
    """Show a sample prediction.

    Parameters
    ----------
    dataset : [type], optional
        [Input dataset, by default None
    num : int, optional
        Number of sample to show, by default 1
    """
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            #t = create_mask(pred_mask)
            display_sample([image[0], mask[0], pred_mask[0]] )
    # else:
    #     # The model is expecting a tensor of the size
    #     # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
    #     # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
    #     # and we want only 1 inference to be faster
    #     # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
    #     one_img_batch = sample_image[0][tf.newaxis, ...]
    #     # one_img_batch -> [1, IMG_SIZE, IMG_SIZE, 3]
    #     inference = model.predict(one_img_batch)
    #     # inference -> [1, IMG_SIZE, IMG_SIZE, N_CLASS]
    #     pred_mask = create_mask(inference)
    #     # pred_mask -> [1, IMG_SIZE, IMG_SIZE, 1]
    #     display_sample([sample_image[0], sample_mask[0],pred_mask[0]])


show_predictions(dataset['val'], 1)