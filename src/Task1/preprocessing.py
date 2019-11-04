import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def read_image(imagename, type):
    return cv2.imread(imagename, type)


def reduce_resolution(image, size):
    dim = (size, size)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    return resized


def display_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(savename, image):
    cv2.imwrite(savename + ".jpg", image)


def scaleradius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def normalization(image):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 100), -4, 128)
    return image


def enhance(image, clip_limit=3):
    # image = cv2.imread(imagepath)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)

    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))

    # convert image from LAB color model back to BGR color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image


image = read_image("../../resources/Task1/Train/original_retinal_images/IDRiD_02.jpg", 1)

# image=reduce_resolution(image,512)
print(image.shape)
image = scaleradius(image, 300)
image = normalization(image)
print(image.shape)
image = enhance(image)
display_image(image)
