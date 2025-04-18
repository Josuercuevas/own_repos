import numpy as np
import cv2
# from jpeg2dct.numpy import load, loads
from matplotlib import pyplot as plt


samples = np.load("generated_images_0722.npy")


def to_img(samples):

    # samples = (sample).permute(0, 2, 3, 1).numpy()

    actual_results = []

    minY = -1024
    minCb = -1016.4375
    minCr = -1017

    maxY = 1016
    maxCb = 1017
    maxCr = 1017

    for image in samples:
        Y = image[:64, :, :]
        Cb = image[64:128, :, :] #Cb
        Cr = image[128:, :, :] #Cr

        # print(f"minY={np.min(Y)}, max={np.max(Y)}")

        # Y_denorm = (((Y+1)*(maxY-minY))/2) + minY
        # Cb_denorm = (((Cb+1)*(maxCb-minCb))/2) + minCb
        # Cr_denorm = (((Cr+1)*(maxCr-minCr))/2) + minCr

        # print(np.min(Y_denorm), np.max(Y_denorm))

        # denorm_image = np.concatenate((Y_denorm, Cb_denorm, Cr_denorm), axis=0)

        # actual_results.append(denorm_image)
        actual_results.append(image)
    
    samples = np.array(actual_results)
    samples = samples.transpose((0, 2, 3, 1))


    dct_img = samples[0, :, :, :]
    dct_y, dct_cb, dct_cr = dct_img[:, :, 0:64], dct_img[:, :, 64:128], dct_img[:, :, 128:192]

    # Y-Channel
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

    # Visualization
    print(img_rec.shape)
    print(img_rec.min(), img_rec.max())
    cv2.imshow("Reconstructed Image", img_rec)
    cv2.waitKey()


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


to_img(samples)

