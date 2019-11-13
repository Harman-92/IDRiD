import numpy as np 
import cv2
import sys
from skimage import exposure
import imageio
from PIL import Image

image_name = "21_training"

train_src = cv2.imread('../../resources/Task2/Training/original_retinal_images/' + image_name + '.tif', cv2.IMREAD_COLOR)
train_mask = np.array(Image.open('../../resources/Task2/Training/background_masks/' + image_name + '_mask.gif'))

# train_mask = cv2.imread('../../resources/Task2/Training/background_masks/' + image_name + '_mask.gif', cv2.IMREAD_GRAYSCALE)
# cv2.imshow('Layer mi', train_mask)

# extract only green from image
train_green = train_src[:,:,1]
# print(train_src.shape, train_green.shape)

# cv2.imshow('Layer 1', train_green)
# cv2.waitKey(0)

# create Gaussian pyramid for function input
# train_pyramid = [train_green]
# mask_pyramid = [train_mask]
# for i in range(3):
#     rows, cols = train_pyramid[i].shape
#     train_pyramid.append(cv2.pyrDown(train_pyramid[i], dstsize=(cols // 2, rows // 2)))
#     mask_pyramid.append(cv2.pyrDown(mask_pyramid[i], dstsize=(cols // 2, rows // 2)))

    # cv2.imshow('Layer %s' % str(i+1), train_pyramid[i])
    # cv2.waitKey(0)

# out_pyramid = []
# for layer in train_pyramid:
# layer = train_pyramid[2]
# mask = mask_pyramid[2]

# currently testing with only input image
layer = train_green
mask = train_mask
print(layer.dtype)
cv2.imshow('Layer a', layer)

# perform histogram stretching
layer = exposure.equalize_hist(layer, mask=mask).astype(np.float32)
print(layer.dtype)
cv2.imshow('Layer b', layer)

# perform bilateral filter
layer = cv2.bilateralFilter(layer,5,150,150)
print(layer.dtype)
cv2.imshow('Layer c', layer)

cv2.waitKey(0)

#create padded input and blank output
layer_out = np.zeros(layer.shape, dtype=np.uint8)

f = layer
(dfx, dfy) = np.gradient(f)
(dfxx, dfxy) = np.gradient(dfx)
(dfyx, dfyy) = np.gradient(dfy)

h,w = layer.shape
for y in range(h):
    for x in range(w):
        hessian = np.array([[dfxx[y,x], dfxy[y,x]], [dfyx[y,x], dfyy[y,x]]])

        # calculate eigenvalues 
        eigens, _ = np.linalg.eig(hessian)
        # print(f, hessian, eigens)
        eigens = abs(eigens)

        # calculate pixel value for output image
        if max(eigens) == 0:
            layer_out[y,x] = 0
        else:
            layer_out[y,x] = (1 - min(eigens) / max(eigens)) * 255

cv2.imshow('Layer', layer_out)
cv2.waitKey(0)

# h,w = layer.shape
# for y in range(h):
#     for x in range(w):
#         # create 3x3 window around each pixel in layer
#         f = layer_in[y:y+3, x:x+3]
#         # print(f)
#         # f = np.random.randint(5, size=(3,3))
#         # print(f.shape)

#         # calculate hessian from gradients
#         (dfx, dfy) = np.gradient(f)
#         (dfxx, dfxy) = np.gradient(dfx)
#         (dfyx, dfyy) = np.gradient(dfy)
#         hessian = np.array([[dfxx[1,1], dfxy[1,1]], [dfyx[1,1], dfyy[1,1]]])
#         # hessian = np.array([[np.mean(dfxx), np.mean(dfxy)], [np.mean(dfyx), np.mean(dfyy)]])

#         if not np.allclose(dfxy,dfyx):
#             print("WAT")
        
#         # calculate eigenvalues 
#         eigens, _ = np.linalg.eig(hessian)
#         # print(f, hessian, eigens)
#         eigens = abs(eigens)

#         # calculate pixel value for output image
#         if max(eigens) == 0:
#             layer_out[y,x] = 0
#         else:
#             layer_out[y,x] = (1 - min(eigens) / max(eigens)) * 255

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