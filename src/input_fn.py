###This script used tf.dataset to define the model input
###Using tf.dataset is always prefered than using placeholder, bc dataset is built directly to tensorflow graph for optimized performance
###using tf.dataset allows batch data to be preloaded.
###learn tf.dataset here https://www.tensorflow.org/guide/datasets

import tensorflow as tf
import os
from io import BytesIO
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa


def process_training_data (file_path, approach=None, split=True):
    train = pd.read_csv(file_path)
    train['Path']='../'+train['Path'].astype(str)
    for i in range(5,19):
        train.iloc[:,i].fillna(value=0, inplace=True)

    X = train.Path.to_numpy()
    y = train.drop(columns=['Path','Sex','Age','Frontal/Lateral','AP/PA'])

    if approach == 'u-Zeroes':
      y.replace(-1.0,0,inplace=True)
    elif approach == 'u-Ones':
      y.replace(-1.0,1.0,inplace=True)
    elif approach == 'u-MultiClass':
      y.replace(-1.0,2.0,inplace=True)
    else:
        pass;

    y = y.to_numpy()
    if split:
      X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.2,
                                                        random_state = 1)

    return X_train, X_test, y_train, y_test



def self_learning_labels(file_path, model_path, split=True):
    #Load the training data
    train = pd.read_csv(file_path)

    for i in range(5,19):
        train.iloc[:,i].fillna(value=0, inplace=True)

    #Load the pre-trained model
    model = load_model(model_path)

    for i in range(len(train['Path'])):
        #Read the image that corresponds to that row in the training data
        image = cv2.imread(train['Path'][i])

        #Get the model's predictions on that image
        predictions = model.predict(image)

        #I am assuming the output of the model is a list/array of 14 integers that correspond to the 14 labels in the training data
        labels = train.iloc[i,5:]
        for j in range(len(labels)):
            #Iterating through every label value in the current row of training data
            #If the value is -1, replace that value with the model's predictions for that label value
            if labels[j] == -1:
                labels[j] = predictions[j]

        #Replace the labels in the training data with the new self-learned labels
        train.iloc[i,5:] = labels

    #Returns the original training dataset with all uncertain label values replaced with self-learned label values
    #Also splits into train and test set if split flag is set to true.
    X = train['Path']
    y = train.iloc[:, 5:]


    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 1)
        return X_train, X_test, y_train, y_test

    return X, y


def process_testing_data(file_path):
    test = pd.read_csv(file_path)
    X = test['Path']
    X = '../'+X.astype(str)
    y = test.iloc[:, 5:]
    return X, y

def process_inference_data(file_path):
    infer = pd.read_csv(file_path)
    X = infer['Path']
    X = '../'+X.astype(str)

    return X


def resize_image_keep_aspect(image,init_width,init_height,MAX_SIZE):
    max = tf.maximum(init_width, init_height)
    ratio = tf.cast(max,dtype=tf.float32)/tf.constant(MAX_SIZE,dtype=tf.float32)
    new_width = tf.cast(tf.cast(init_width,dtype=tf.float32) /ratio,dtype=tf.int32)
    new_height = tf.cast(tf.cast(init_height,dtype=tf.float32) /ratio,dtype=tf.int32)
    return tf.image.resize(image,[new_width,new_height])

###parse_fucntion for saving the trained servable
def parse_function_serve(base64Img):
    image_string=tf.decode_base64(base64Img)
    image=tf.image.decode_jpeg(image_string, channels=3)
    shape = tf.shape(image)
    init_width = shape[0]
    init_height = shape[1]
    max_size = 320
    resized_image = resize_image_keep_aspect(image,init_width,init_height,max_size)
    image_padded = tf.image.resize_with_crop_or_pad(resized_image,max_size,max_size)
    #final_image_padded=tf.image.convert_image_dtype(image_padded,dtype=tf.float32)
    final_image_padded /= 255.0
    return final_image_padded

def parse_function(filename, label):
    image_string=tf.io.read_file(filename)
    image=tf.image.decode_jpeg(image_string, channels=3)
    shape = tf.shape(image)
    init_width = shape[0]
    init_height = shape[1]
    max_size = 320
    resized_image = resize_image_keep_aspect(image,init_width,init_height,max_size)
    image_padded = tf.image.resize_with_crop_or_pad(resized_image,max_size,max_size)
    final_image_padded=tf.image.convert_image_dtype(image_padded,dtype=tf.float32)
    final_image_padded /= 255.0
    return final_image_padded,label

def parse_function_test(filename):
    image_string=tf.io.read_file(filename)
    image=tf.image.decode_jpeg(image_string, channels=3)
    shape = tf.shape(image)
    init_width = shape[0]
    init_height = shape[1]
    max_size = 320
    resized_image = resize_image_keep_aspect(image,init_width,init_height,max_size)
    image_padded = tf.image.resize_with_crop_or_pad(resized_image,max_size,max_size)
    final_image_padded=tf.image.convert_image_dtype(image_padded,dtype=tf.float32)
    final_image_padded /= 255.0
    return final_image_padded

def image_augmentation(image, label):
    random_angle = tf.random.uniform(shape = [1], minval = -np.pi / 4, maxval = np.pi / 4)
    image = tfa.image.transform_ops.rotate(image, random_angle)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.7, 1.3)
    image = tf.image.random_saturation(image, 0.6, 1.6)
    return image, label

####funuction for creating tensorflow dataset.
####resize image with aspect ration
####image augmentation
def input_fn_multi_output(is_training, filenames,labels,batch_size):
    num_samples=len(filenames)
    #assert len(filenames)==len(labels), "Filenames and labels should have same length"
    if labels is None:
        inputs = tf.constant(filenames)
        dataset = (tf.data.Dataset.from_tensor_slices(inputs)
            .map(parse_function_test,num_parallel_calls=4)
            .apply(tf.data.experimental.ignore_errors())
            .batch(batch_size)
            .prefetch(5)
        )
        return dataset

    else:

        Ys = []
        nums = labels.shape[1]
        for i in range(nums):
            Ys.append(list(labels[:,i]))
        Ys = tuple(Ys)


        x = tf.data.Dataset.from_tensor_slices(filenames)
        ys = tf.data.Dataset.from_tensor_slices(Ys)

        if is_training:
            dataset=(tf.data.Dataset.zip((x, ys))
                .shuffle(num_samples)
                .map(parse_function,num_parallel_calls=4)
                .apply(tf.data.experimental.ignore_errors())
                .map(image_augmentation,num_parallel_calls=4)
                .repeat()
                .batch(batch_size)
                .prefetch(5)
            )
        else:
            dataset = (tf.data.Dataset.zip((x, ys))
                .map(parse_function,num_parallel_calls=4)
                .apply(tf.data.experimental.ignore_errors())
                .repeat()
                .batch(batch_size)
                .prefetch(5)
            )

        return dataset

def compute_weights(train_y):
    result = []
    nums = train_y.shape[1]
    for i in range(nums):
         unique, counts = np.unique(train_y[:,i], return_counts=True)
         result.append(dict(zip(unique, counts)))
    return result
