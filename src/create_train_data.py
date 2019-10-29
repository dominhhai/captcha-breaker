from __future__ import print_function, absolute_import, division
import os
import json
from skimage import io
from skimage.color import rgb2gray
from img import split_letters
import numpy as np


# Helper methods
def strip_extension(filename):
    return filename[:filename.rindex('.')]


def build_data_map(data_path):
    files = os.listdir(data_path)
    return {x: strip_extension(x) for x in files}


DATA_DIR = 'data'
DATA_FULL_DIR = os.path.join(DATA_DIR, 'captcha')
DATA_TRAIN_DIR = os.path.join(DATA_DIR, 'train')
DATA_TRAIN_FILE = os.path.join(DATA_DIR, 'captcha')

# array of tuple of binary image and label
data_x = []
data_y = []

# build image contents map
image_contents = build_data_map(DATA_FULL_DIR)

# load image and save letters
counter = 0

for fname, contents in image_contents.iteritems():
    counter += 1
    print(counter, fname, contents)
    original_image = io.imread(os.path.join(DATA_FULL_DIR, fname))
    grayscale_image = rgb2gray(original_image)

    # split image
    letters = split_letters(grayscale_image, num_letters=len(contents), debug=True)
    if letters != None:
        for i, letter in enumerate(letters):
            content = contents[i]
            # add to dataset
            data_x.append(letter)
            data_y.append(np.uint8(ord(content) - 65)) # 65: 'A'

            # save letter into train folder
            fpath = os.path.join(DATA_TRAIN_DIR, content)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            letter_fname = os.path.join(fpath, str(i+1) + '-' + strip_extension(fname) + '.png')
            io.imsave(letter_fname, 255 - letter) # invert black <> white color
    else:
        print('Letters is not valid')
        break

# split into train and test data set
train_num = int(len(data_y) * 0.8) # 80%

# save train data
print('saving dataset')
np.savez_compressed(DATA_TRAIN_FILE,
    x_train=data_x[:train_num], y_train=data_y[:train_num],
    x_test=data_x[train_num:], y_test=data_y[train_num:])
