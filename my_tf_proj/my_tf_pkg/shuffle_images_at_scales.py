import os, sys, tarfile
import urllib.request as urllib
import numpy as np
import matplotlib.pyplot as plt
import pdb

def read_single_image(arg,image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    # IMG_SIZE = HEIGHT * WIDTH * DEPTH
    image_array_bytes = np.fromfile(image_file, dtype=np.uint8, count=arg.IMG_SIZE)
    # force into image tensor
    image_tensor = np.reshape(image_array_bytes, (3, 96, 96)) # (chan, D, D)
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like their channels separated.
    image_standard_form = np.transpose(image_tensor, (2, 1, 0)) # (D, D, chan)
    return image_standard_form

#

def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """
    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8) # array of all bytes representing the image

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96)) # (N, chan, D, D)

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1)) # (N, chan, D, D)
        return images

#

def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()

#

def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels

# helper function

def read_single_image_as_bytes_array():
    '''
    get an image as a bytes array from a file.
    '''
    # read a single image, count determines the number of uint8's to read
    # IMG_SIZE = HEIGHT * WIDTH * DEPTH
    image = np.fromfile(image_file, dtype=np.uint8, count=arg.IMG_SIZE)
    print(image)
    print(type(image))
    print(image.shape)
    print(max(image))
    return image

# extra functions

def show_file_reader_position(f):
    '''
    show the position of the file reader.
    '''
    print('f.tell() ', f.tell())

def plot_a_single_image(i,images):
    '''
    assumes data set is in format (N,D,D,C)
    '''
    plot_image( images[i,:,:,:] )

def params_from_stl10():
    # image shape
    HEIGHT, WIDTH, DEPTH = 96, 96, 3

    # size of a single image in bytes
    IMG_SIZE = HEIGHT * WIDTH * DEPTH

    # path to the directory with the data
    DATA_DIR = './data'
    # url of the binary data
    DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    # path to the binary train file with image data
    DATA_PATH = './data/stl10_binary/train_X.bin'
    # path to the binary train file with labels
    LABEL_PATH = './data/stl10_binary/train_y.bin'
