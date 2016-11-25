import os, sys, tarfile
#import urllib.request, urllib.parse, and urllib.error
import urllib.request as urllib
import numpy as np
import matplotlib.pyplot as plt

import my_tf_pkg.shuffle_images_at_scales as shuf

import namespaces as ns

import pdb

def get_arg():
    '''
    gets size of images, paths etc. in a namespace arg.
    '''
    # image shape
    HEIGHT, WIDTH, DEPTH = 96, 96, 3
    # size of a single image in bytes
    IMG_SIZE = HEIGHT * WIDTH * DEPTH
    # path to the directory with the data
    DATA_DIR = './data'
    # path to the binary train file with image data
    DATA_PATH = './data/stl10_binary/train_X.bin'
    # path to the binary train file with labels
    LABEL_PATH = './data/stl10_binary/train_y.bin'
    # load to namespace
    arg = ns.Namespace(DATA_DIR=DATA_DIR,DATA_PATH=DATA_PATH,LABEL_PATH=LABEL_PATH)
    arg.HEIGHT, arg.WIDTH, arg.DEPTH = HEIGHT, WIDTH, DEPTH
    arg.IMG_SIZE = IMG_SIZE
    return arg

def test_show_one_image(arg):
    ## test to check if the image is read correctly
    with open(arg.DATA_PATH) as f:
        shuf.show_file_reader_position(f)
        image = shuf.read_single_image(arg,f)
        shuf.plot_image(image)
        shuf.show_file_reader_position(f)

if __name__ == "__main__":
    arg = get_arg()
    ## test to check if the image is read correctly
    #test_show_one_image(arg)

    # test to check if the whole dataset is read correctly
    images = shuf.read_all_images(arg.DATA_PATH)
    print( images.shape )
    shuf.plot_a_single_image(2,images)

    labels = shuf.read_labels(arg.LABEL_PATH)
