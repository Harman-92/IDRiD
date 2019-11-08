import numpy as np 
import cv2
import sys

image_name = "21_training"

train_src = cv2.imread('../../resources/Task2/Training/original_retinal_images/' + image_name + '.tif', cv2.IMREAD_COLOR)
train_mask = cv2.imread('../../resources/Task2/Training/background_masks/' + image_name + '_mask.tif', cv2.IMREAD_UNCHANGED)

# extract only green from image
train_green = train_src[:,:,1]

cv2.imshow('Layer 1', train_green)
cv2.waitKey(0)

# create Gaussian pyramid for function input
train_pyramid = [train_green]
for i in range(3):
    rows, cols = train_pyramid[i].shape
    train_pyramid.append(cv2.pyrDown(train_pyramid[i], dstsize=(cols // 2, rows // 2)))

    cv2.imshow('Layer %s' % str(i+1), train_pyramid[i])
    cv2.waitKey(0)

out_pyramid = []
for layer in train_pyramid:
    #create padded input and blank output
    layer_in = cv2.copyMakeBorder(layer,1,1,1,1,cv2.BORDER_DEFAULT)
    layer_out = np.zeros(layer.shape)

    h,w = layer.shape
    for y in range(h):
        for x in range(w):
            # create 3x3 window around each pixel in layer
            window = layer_in[y:y+2, x:x+2]

            #TODO: calculate hessian matrix for window
            
            #TODO: calculate eigenvalues 

            #TODO: calculate pixel value for output image
            # if eig_1 > eig_2:
            #     layer_out[y,x] = 1 - eig_1/eig_2
            # else:
            #     layer_out[y,x] = 1 - eig_2/eig_1

# train_out = np.zeros(train_green.shape)
# # iterate through each image in out_pyramid
# for layer in out_pyramid:
#     # upscale layer to original size
#     if layer.shape != train_green.shape:
#         layer = cv2.resize(layer, train_green.shape)

#     # apply hysteresis to get binary image
#     ret, thresh = cv2.threshold(layer, 217, 237, cv2.THRESH_BINARY) 

#     # combine all images to create final image
#     train_out = cv2.bitwise_and(train_out, thresh)