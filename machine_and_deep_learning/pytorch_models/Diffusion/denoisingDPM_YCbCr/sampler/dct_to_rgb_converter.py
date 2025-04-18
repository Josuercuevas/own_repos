import numpy as np
import cv2
from matplotlib import pyplot as plt

# Denormalizes the values in each component
def denormalize(array, minVal, maxVal):
    return ((array + 1) * (maxVal - minVal) / 2) + minVal


# Handles converting the array to RGB
def dct_2_rgb(dct_y, dct_cb, dct_cr):
    dct_y = dct_y.transpose((1, 2, 0))
    dct_cb = dct_cb.transpose((1, 2, 0))
    dct_cr = dct_cr.transpose((1, 2, 0))

    # Y channel
    rows, cols, _ = dct_y.shape
    imgY_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_y[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgY_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    imgY_rec[imgY_rec < 0] = 0
    imgY_rec[imgY_rec > 255] = 255
    imgY_rec = np.uint8(imgY_rec)

    # Cb-Channel
    rows, cols, _ = dct_cb.shape
    # print('dct_cb shape = ', dct_y.shape)
    imgCb_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_cb[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgCb_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    imgCb_rec[imgCb_rec < 0] = 0
    imgCb_rec[imgCb_rec > 255] = 255
    imgCb_rec = np.uint8(imgCb_rec)

    # Cr-Channel
    rows, cols, _ = dct_cr.shape
    imgCr_rec = np.ones(shape=(8*rows, 8*cols))
    for j in range(rows):
        for i in range(cols):
            spectrogram = reshape(dct_cr[j, i])
            macroblock = cv2.idct(spectrogram) + 128
            imgCr_rec[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = macroblock
    imgCr_rec[imgCr_rec < 0] = 0
    imgCr_rec[imgCr_rec > 255] = 255
    imgCr_rec = np.uint8(imgCr_rec)
    img_rec = np.dstack((imgY_rec, imgCr_rec, imgCb_rec))
    img_rec = np.uint8(img_rec)
    img_rec = cv2.cvtColor(img_rec, cv2.COLOR_YCrCb2BGR)
    img_rec[img_rec < 0] = 0
    img_rec[img_rec > 255] = 255

    return img_rec


# Handles the extraction of the components, 
# the de-normalization steps, conversion to image, and returns it
def to_img(image):
    minY = -1024
    minCb = -993
    minCr = -891
    maxY = 1016
    maxCb = 980
    maxCr = 1034

    Y_norm = image[:64, :, :] #Y
    Cb_norm = image[64:128, :, :] #Cb
    Cr_norm = image[128:, :, :] #Cr

    Y = denormalize(Y_norm, minY, maxY)
    Cb = denormalize(Cb_norm, minCb, maxCb)
    Cr = denormalize(Cr_norm, minCr, maxCr)

    image = dct_2_rgb(Y,Cb,Cr)
    return image


# Necessary to transform into a 2D array (for the components of DCT)
def reshape(arr):
    FLATTEN_KEY = np.arange(64).reshape((8, 8))
    arr = np.asarray(arr)
    if arr.shape != (64,):
        print("array shape --> ", arr.shape)
        raise ValueError('Array needs to be macroblock of shape 8x8')
    rows = cols = 8
    macroblock = np.empty(shape=(rows, cols))
    for j in range(rows):
        for i in range(cols):
            idx = FLATTEN_KEY[j, i]
            macroblock[j, i] = arr[idx]
    return macroblock
