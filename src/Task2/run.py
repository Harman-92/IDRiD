import numpy as np 
import cv2
import sys
from skimage import exposure
import imageio
from PIL import Image
from math import pow, sqrt, exp
import matplotlib.pyplot as plt

NUM_LAYERS = 3


parameters = {
    "0": {
        "b": 0.05,
        "c": 0.04,
        "h_min": 30,
        "h_max": 70,
        "mode": 0
    },
    "1": {
        "b": 0.05,
        "c": 0.04,
        "h_min": 30,
        "h_max": 70,
        "mode": 1
    },
    "2": {
        "b": 0.05,
        "c": 0.04,
        "h_min": 30,
        "h_max": 70,
        "mode": 2
    }
}


def hysteresis_thresholding(layer, h_min, h_max, value):
    layer_pad = cv2.copyMakeBorder(layer, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    h, w = layer.shape
    layer_out = np.zeros(layer.shape)
    for y in range(h):
        for x in range(w):
            # create 3x3 window around each pixel in layer
            f = layer_pad[y:y + 3, x:x + 3]
            if layer[y, x] >= h_max:
                layer_out[y, x] = value
            elif h_min <= layer[y, x] < h_max:
                gt = f.__ge__(h_max)
                if gt.any():
                    layer_out[y, x] = value
    return layer_out


def get_masked_output(input, mask):
    try:
        return cv2.bitwise_and(src1=input.astype(np.float32), src2=mask.astype(np.float32))
    except Exception as e:
        print(f'Masking failed due to: {e}')
        return input


def hessian_analysis(layer):
    # create padded input and blank output
    layer_out = np.zeros(layer.shape, dtype=np.uint8)

    f = layer
    (dfx, dfy) = np.gradient(f)
    (dfxx, dfxy) = np.gradient(dfx)
    (dfyx, dfyy) = np.gradient(dfy)

    h, w = layer.shape
    for y in range(h):
        for x in range(w):
            hessian = np.array([[dfxx[y, x], dfxy[y, x]], [dfyx[y, x], dfyy[y, x]]])

            # calculate eigen values
            eigens, _ = np.linalg.eig(hessian)
            eigens = abs(eigens)

            # calculate pixel value for output image
            if max(eigens) == 0:
                layer_out[y, x] = 0
            else:
                e_min = min(eigens)
                e_max = max(eigens)
                Rb = e_min / e_max
                S = sqrt(pow(e_min, 2) + pow(e_max, 2))
                b = parameters[str(i)]["b"]
                c = parameters[str(i)]["c"]
                # Vesselness feature
                v_feature = exp(-1 * (pow(Rb, 2) / (2 * pow(b, 2)))) * (1 - exp(-1 * (pow(S, 2) / (2 * pow(c, 2)))))
                # layer_out[y,x] = (1 - min(eigens) / max(eigens)) * 255
                layer_out[y, x] = v_feature * 255

    return layer_out


def resize_image(input, reference):
    output = input
    if input.shape != reference.shape:
        output = cv2.resize(input, (reference.shape[1], reference.shape[0]))
    return output


def apply_morphological_postprocessing(input, mode):
    output = input
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    if mode == 0:
        #output = cv2.erode(input, kernel2, iterations=1)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel2)
    else:
        output = cv2.erode(input, kernel2, iterations=1)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel2)
    return output


    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)


image_name = "21_training"

train_src = cv2.imread('../../resources/Task2/Training/original_retinal_images/' + image_name + '.tif', cv2.IMREAD_COLOR)
train_mask = np.array(Image.open('../../resources/Task2/Training/background_masks/' + image_name + '_mask.gif'))

# extract only green from image
train_green = train_src[:, :, 1]

# create Gaussian pyramid for function input
train_pyramid = [train_green]
mask_pyramid = [train_mask]

for i in range(NUM_LAYERS):
    rows, cols = train_pyramid[i].shape
    train_pyramid.append(cv2.pyrDown(train_pyramid[i], dstsize=(cols // 2, rows // 2)))
    # mask_pyramid.append(cv2.pyrDown(mask_pyramid[i], dstsize=(cols // 2, rows // 2)))
    #TODO fix this so mask doesn't interpolate
    mask_pyramid.append(cv2.resize(mask_pyramid[i], (cols // 2, rows // 2)))

fig = plt.figure()
fig.suptitle("Layers")
out_pyramid = []

for i in range(NUM_LAYERS):
    layer = train_pyramid[i].astype(np.float32)
    mask = mask_pyramid[i]

    # perform histogram stretching
    layer = exposure.equalize_hist(layer, mask=mask).astype(np.float32)

    # perform bilateral filter
    layer = cv2.bilateralFilter(layer, 4, 150, 150)

    # perform hessian analysis
    layer_out = hessian_analysis(layer)

    # resize pyramid layers to original image size
    layer_out = resize_image(layer_out, train_green)

    # apply hysteresis thresholding
    layer_thresh = hysteresis_thresholding(layer_out, parameters[str(i)]["h_min"], parameters[str(i)]["h_max"], 255)

    #ret, thresh = cv2.threshold(layer_out, 217, 255, cv2.THRESH_BINARY)

    # apply masking
    layer_thresh = get_masked_output(layer_thresh, train_mask)

    layer_thresh = apply_morphological_postprocessing(layer_thresh, parameters[str(i)]["mode"])


    # thresh = cv2.bitwise_and(src1=thresh.astype(np.float32), src2=train_mask.astype(np.float32))
    # layer_thresh = cv2.bitwise_and(src1=layer_thresh.astype(np.float32), src2=train_mask.astype(np.float32))

    fig.add_subplot(2, 3, i+1)
    plt.imshow(layer_out, cmap='gray')
    plt.axis('off')
    plt.title('No thresh')

    fig.add_subplot(2, 3, i+4)
    plt.imshow(layer_thresh, cmap='gray')
    plt.axis('off')
    plt.title('Thresh')

    print('Completed Layer %s' % i)
    out_pyramid.append(layer_thresh)

plt.show()

train_out = np.zeros(train_green.shape, dtype=np.float32)
# iterate through each image in out_pyramid
for i in range(NUM_LAYERS):
    layer = out_pyramid[i]

    # combine all images to create final image
    train_out = cv2.bitwise_or(src1=train_out, src2=layer)
cv2.imshow('Output', train_out)

cv2.waitKey(0)