import keras
from keras.datasets import mnist

import numpy as np
from PIL import Image, ImageOps
import os


def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)


# Load MNIST Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

DIR_NAME = "JPEGImages"
if os.path.exists(DIR_NAME) == False:
    os.mkdir(DIR_NAME)
DIR_TRAIN = DIR_NAME + '/train/'
DIR_TEST = DIR_NAME + '/test/'
if os.path.exists(DIR_TRAIN) == False:
    os.mkdir(DIR_TRAIN)
if os.path.exists(DIR_TEST) == False:
    os.mkdir(DIR_TEST)

# Save Images
# i = 0
# for li in [x_train, x_test]:
#     print("[---------------------------------------------------------------]")
#     for x in li:
#         filename = "{0}/{1:05d}.jpg".format(DIR_NAME,i)
#         print(filename)
#         save_image(filename, x)
#         i += 1

for (i, x) in enumerate(x_train):
    filename = '{0}{1:01d}_{2:05d}.jpg'.format(DIR_TRAIN, y_train[i], i)
    save_image(filename, x)
for (i, x) in enumerate(x_test):
    filename = '{0}{1:01d}_{2:05d}.jpg'.format(DIR_TEST, y_test[i], i)
    save_image(filename, x)
