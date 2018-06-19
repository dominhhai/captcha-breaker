from __future__ import print_function, absolute_import, division
from math import floor, ceil
from skimage import img_as_ubyte
from skimage.measure import find_contours
from skimage.util import crop
from skimage.transform import resize

import matplotlib.pyplot as plt

SHIFT_PIXEL = 10 # shift image from right to left
BINARY_THRESH = 30 # image binary thresh
LETTER_SIZE = (36, 36) # letter width, heigth

def split_letters(image, num_letters=6, debug=False):
    '''
    split full captcha image into `num_letters` lettersself.
    return list of letters binary image (0: white, 255: black)
    '''
    # move left
    left = crop(image, ((0, 0), (0, image.shape[1]-SHIFT_PIXEL)), copy=True)
    image[:,:-SHIFT_PIXEL] = image[:,SHIFT_PIXEL:]
    image[:,-SHIFT_PIXEL:] = left
    # binarization
    binary = image > BINARY_THRESH
    # find contours
    contours = find_contours(binary, 0.5)
    contours = [[
        [int(floor(min(contour[:, 1]))), int(floor(min(contour[:, 0])))], # top-left point
        [int(ceil(max(contour[:, 1]))), int(ceil(max(contour[:, 0])))]  # down-right point
      ] for contour in contours]
    # keep letters order
    contours = sorted(contours, key=lambda contour: contour[0][0])
    # find letters box
    letter_boxs = []
    for contour in contours:
        if len(letter_boxs) > 0 and contour[0][0] < letter_boxs[-1][1][0] - 5:
            # skip inner contour
            continue
        # extract letter boxs by contour
        boxs = get_letter_boxs(binary, contour)
        for box in boxs:
            letter_boxs.append(box)
    # check letter outer boxs number
    if len(letter_boxs) != num_letters:
        print('ERROR: number of letters is NOT valid', len(letter_boxs))
        # debug
        if debug:
            print(letter_boxs)
            plt.imshow(binary, interpolation='nearest', cmap=plt.cm.gray)
            for [x_min, y_min], [x_max, y_max] in letter_boxs:
                plt.plot(
                    [x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    linewidth=2)
            plt.xticks([])
            plt.yticks([])
            plt.show()
        return None

    # normalize size (40x40)
    letters = []
    for [x_min, y_min], [x_max, y_max] in letter_boxs:
        letter = resize(image[y_min:y_max, x_min:x_max], LETTER_SIZE)
        letter = img_as_ubyte(letter < 0.6)
        letters.append(letter)

    return letters

def get_letter_boxs(binary, contour):
    boxs = []
    w = contour[1][0] - contour[0][0] # width
    h = contour[1][1] - contour[0][1] # height
    if w < 10:
        # skip too small contour (noise)
        return boxs

    if w < 37 and w / h < 1.1:
        boxs.append(contour)
    else:
        # split 2 letters if w is large
        x_mean = contour[0][0] + int(round(w / 2))
        sub_contours = [
            [contour[0], [x_mean, contour[1][1]]],
            [[x_mean, contour[0][1]], contour[1]]
        ]
        for [x_min, y_min], [x_max, y_max] in sub_contours:
            # fit y_min, y_max
            y_min_val = min(binary[y_min + 1, x_min:x_max])
            y_max_val = min(binary[y_max - 1, x_min:x_max])
            while y_min_val or y_max_val:
                if y_min_val:
                    y_min += 1
                    y_min_val = min(binary[y_min + 1, x_min:x_max])
                if y_max_val:
                    y_max -= 1
                    y_max_val = min(binary[y_max - 1, x_min:x_max])

            boxs.append([[x_min, y_min], [x_max, y_max]])

    return boxs
