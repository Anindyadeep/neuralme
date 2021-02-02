############################## PREPARATION OF THE IMAGE DATASETS ##############################

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import image
import streamlit as st
import tensorflow as tf  # only to get the numpy array from the images


def get_the_array_from_image_train(directory, eg, label):
    my_list = []
    my_list_arr = []  # final list to be put into the array

    # Parsing the folders to get the image

    for filename in os.listdir(directory):
        img_temp_path = os.path.join(directory, filename)
        temp_img = Image.open(img_temp_path)
        temp_img = temp_img.resize((64, 64))
        arr = tf.keras.preprocessing.image.img_to_array(temp_img)
        my_list.append(arr)

    # Resolving the shape problem in the numpy array as all the images are not of dim = 64 x 64 x 1

    for i in range(eg):
        if my_list[i].shape[2] == 1:
            my_list_arr.append(my_list[i])

    # Appending the label at the last of the reqiured numpy array

    Nd_arr = np.array(my_list_arr)
    Nd_arr = Nd_arr.reshape(Nd_arr.shape[0], 4096)

    # After resolving the problem putting all the things into the final numpy array

    Nd_arr_train = np.zeros(shape=(Nd_arr.shape[0], 4097))

    for i in range(Nd_arr.shape[0]):
        var = Nd_arr[i]
        var = np.append(var, label)
        Nd_arr_train[i] = var

    return Nd_arr_train


def visualise_plots(directory, stream = False):
    List_dir = []
    count = 0

    # getting the images from the directory

    for filename in os.listdir(directory):
        img_temp_path = os.path.join(directory, filename)
        List_dir.append(img_temp_path)
        count += 1
        if (count > 16):
            break

    # The number of plots and the plotting

    wide = 10
    height = 10
    fig = plt.figure(figsize=(9, 9))
    plt.axis('off')
    rows = 4
    columns = 4

    for i in range(1, columns * rows + 1):
        img = image.imread(List_dir[i])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)

    if stream:
        st.pyplot(plt)
    else:
        plt.show()
