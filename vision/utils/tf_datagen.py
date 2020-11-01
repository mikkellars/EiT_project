"""[summary]
Python class to make data_generators for ada20k dataset
Inspiration from https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/
"""

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

class gen_fence():

    def __init__(self, dataset_path, train_folder, val_folder, batch_size:int = 1, img_size:int = 512, n_channels:int = 3):
        self.img_size = img_size
        self.n_channels = n_channels
        self.n_classes = 2
        print(f"The dataset has {self.n_classes} classes")
        self.train_data_size =  len(glob.glob(dataset_path + train_folder + "*.jpg"))
        self.val_data_size =  len(glob.glob(dataset_path + val_folder + "*.jpg"))
        print(f"The Training Dataset contains {self.train_data_size} images.")
        print(f"The Validation Dataset contains {self.val_data_size} images.")

        self.__parse_image("/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/train_set/labels/training/2017_Train_00001.png")

        # -- Train Dataset --#
        self.train_dataset = tf.data.Dataset.list_files(dataset_path + train_folder + "*.jpg")
        self.train_dataset = self.train_dataset.map(self.__parse_image)
        self.train_dataset = self.train_dataset.map(self.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.train_dataset = self.train_dataset.shuffle(buffer_size=10)
        self.train_dataset = self.train_dataset.repeat()
        self.train_dataset = self.train_dataset.batch(batch_size)
        self.train_dataset = self.train_dataset.prefetch(buffer_size=AUTOTUNE)

        #-- Validation Dataset --#
        self.val_dataset = tf.data.Dataset.list_files(dataset_path + val_folder + "*.jpg")
        self.val_dataset = self.val_dataset.map(self.__parse_image)
        self.val_dataset = self.val_dataset.map(self.load_image_test)
        self.val_dataset = self.val_dataset.repeat()
        self.val_dataset = self.val_dataset.batch(batch_size)
        self.val_dataset = self.val_dataset.prefetch(buffer_size=AUTOTUNE)


    # -------------------
    # Public functions 
    # -------------------

    def display_sample(self, display_list):
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


    def display_sample_opencv(self, img, gr_mask, pr_mask=None):
        """Show side-by-side an input image,
        the ground truth and the prediction.
        Does not work yet
        """
        img_plot = tf.keras.preprocessing.image.array_to_img(img)
        gr_mask_plot = tf.keras.preprocessing.image.array_to_img(gr_mask)

        # Initialize a list of colors to represent each class label
        class_names = []
        for cl in self.min_dataset_classes:
            class_names.append(cl)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(self.n_classes, 3), dtype="uint8")
        colors = np.vstack([[0, 0, 0], colors]).astype("uint8") # add black to backround color infront of colors

        # Initialize the legend visualization
        legend = np.zeros(((self.n_classes * 25) + 25, 300, 3), dtype="uint8")
        # Loop over the class names + colors
        for (i, (className, color)) in enumerate(zip(class_names, colors)):
            # draw the class name + color on the legend
            color = [int(c) for c in color]
            cv2.putText(legend, className, (5, (i * 25) + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
                tuple(color), -1)

        t = np.array(gr_mask_plot)
        img_plot = cv2.cvtColor(np.array(img_plot), cv2.COLOR_RGB2BGR)
        gr_mask_plot = colors[np.array(gr_mask_plot)]
        gr_mask_plot = cv2.cvtColor(np.array(gr_mask_plot), cv2.COLOR_RGB2BGR)

        cv2.imshow("Legend", legend)
        cv2.imshow("Input", img_plot)
        cv2.imshow("Output", gr_mask_plot)
        cv2.waitKey(0)

        # if pr_mask == None:
        #     title = ['Input Image', 'True Mask', 'Predicted Mask']

        #     for i in range(len(display_list)):
        #         plt.subplot(1, len(display_list), i+1)
        #         plt.title(title[i])
        #         plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        #         plt.axis('off')
        #     plt.show()

        # else:

    def get_datasets(self):
        dataset = {"train": self.train_dataset, "val": self.val_dataset}
        return dataset

    # -------------------
    # Private functions 
    # -------------------

    def __channel_mask(self, mask):
        """Converts from 1 channel with a label, to each number of channels representing a class

        Args:
            mask ([type]): [description]
        """
        class_labels = list(range(0, self.n_classes))
        one_hot_map = list()

        for cl in class_labels:
            cl_map = tf.reduce_all(tf.equal(mask, cl), axis=-1)
            one_hot_map.append(cl_map)
            
        one_hot_map = tf.stack(one_hot_map, axis=-1)
        one_hot_map = tf.cast(one_hot_map, tf.float32)

        return one_hot_map

    def __parse_image(self, img_path: str) -> dict:
        """Load an image and its annotation (mask) and returning
        a dictionary.

        Parameters
        ----------
        img_path : str
            Image (not the mask) location.

        Returns
        -------
        dict
            Dictionary mapping an image and its annotation.
        """
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.uint8)

        # For one Image path:
        # .../trainset/images/training/ADE_train_00000001.jpg
        # Its corresponding annotation path is:
        # .../trainset/annotations/training/ADE_train_00000001.png
        mask_path = tf.strings.regex_replace(img_path, "images", "labels")
        mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
        mask = tf.io.read_file(mask_path)
        # The masks contain a class index for each pixels
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.uint8)
        tf.where(mask == 255, np.dtype('uint8').type(1), mask) # Make 255 value to 1

       # mask_new = self.__minimize_mask(mask)
       # mask_new = self.__channel_mask(mask) # Make each label to each own channel
        return {'image': image, 'segmentation_mask': mask}

    @tf.function
    def normalize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
        """Rescale the pixel values of the images between 0.0 and 1.0
        compared to [0,255] originally.

        Parameters
        ----------
        input_image : tf.Tensor
            Tensorflow tensor containing an image of size [SIZE,SIZE,3].
        input_mask : tf.Tensor
            Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

        Returns
        -------
        tuple
            Normalized image and its annotation.
        """
        input_image = tf.cast(input_image, tf.float32) / 255.0
        return input_image, input_mask

    @tf.function
    def load_image_train(self, datapoint: dict) -> tuple:
        """Apply some transformations to an input dictionary
        containing a train image and its annotation.

        Notes
        -----
        An annotation is a regular  channel image.
        If a transformation such as rotation is applied to the image,
        the same transformation has to be applied on the annotation also.

        Parameters
        ----------
        datapoint : dict
            A dict containing an image and its annotation.

        Returns
        -------
        tuple
            A modified image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (self.img_size, self.img_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.img_size, self.img_size))

        if tf.random.uniform(()) > 0.5:
            input_image = tf.image.flip_left_right(input_image)
            input_mask = tf.image.flip_left_right(input_mask)

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

    @tf.function
    def load_image_test(self, datapoint: dict) -> tuple:
        """Normalize and resize a test image and its annotation.

        Notes
        -----
        Since this is for the test set, we don't need to apply
        any data augmentation technique.

        Parameters
        ----------
        datapoint : dict
            A dict containing an image and its annotation.

        Returns
        -------
        tuple
            A modified image and its annotation.
        """
        input_image = tf.image.resize(datapoint['image'], (self.img_size, self.img_size))
        input_mask = tf.image.resize(datapoint['segmentation_mask'], (self.img_size, self.img_size))

        input_image, input_mask = self.normalize(input_image, input_mask)

        return input_image, input_mask

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
    pred_mask = tf.argmax(tf.squeeze(pred_mask), axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
        
DATA_PATH = '/home/mikkel/Documents/experts_in_teams_proj/vision/data/fence_data/train_set'
datagenerator = gen_fence(DATA_PATH, '/images/training/', '/images/validation/')
dataset = datagenerator.get_datasets()
for image, mask in dataset['val'].take(1):
    sample_image, sample_mask = image, mask#create_mask(mask)

datagenerator.display_sample((sample_image[0], sample_mask[0]))

