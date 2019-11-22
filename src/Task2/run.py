import numpy as np 
import cv2
import sys
from skimage import exposure
import imageio
from PIL import Image
from math import pow, sqrt, exp
import matplotlib.pyplot as plt
import json

PARAMETERS = json.load(open('parameters.json'))
NUM_LAYERS = len(PARAMETERS)


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
                b = PARAMETERS[str(i)]["b"]
                c = PARAMETERS[str(i)]["c"]
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


def gaussian_pyramid(image, mask, num):
    train_pyramid = [image]
    mask_pyramid = [mask]
    for i in range(num):
        rows, cols = train_pyramid[i].shape
        train_pyramid.append(cv2.pyrDown(train_pyramid[i], dstsize=(cols // 2, rows // 2)))
        # mask_pyramid.append(cv2.pyrDown(mask_pyramid[i], dstsize=(cols // 2, rows // 2)))
        # TODO fix this so mask doesn't interpolate
        mask_pyramid.append(cv2.resize(mask_pyramid[i], (cols // 2, rows // 2)))
    return train_pyramid, mask_pyramid


def apply_morphological_postprocessing(input, mode):
    output = input
    kernel = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((3, 3), np.uint8)
    kernel3 = np.ones((2, 2), np.uint8)

    if mode == 0:
        #output = cv2.erode(input, kernel3, iterations=1)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
        #output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    elif mode == 1:
        #output = cv2.erode(output, kernel2, iterations=1)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel2)
    elif mode == 2:
        output = cv2.erode(output, kernel2, iterations=1)
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)

    else:
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel3)
    return output


def get_metrics(pred, gt):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    if pred.shape == gt.shape:
        h, w = layer.shape
        for y in range(h):
            for x in range(w):
                if pred[y][x] == gt[y][x]:
                    if pred[y][x] == 255:
                        tp += 1
                    else:
                        tn += 1
                else:
                    if pred[y][x] == 255:
                        fp += 1
                    else:
                        fn += 1
    else:
        print('Size does not match')

    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
    sensitivity = (tp / (tp + fn)) * 100
    specificity = (tn / (tn + fp)) * 100

    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"Accuracy: {accuracy}%")
    print(f"Sensitivity: {sensitivity}%")
    print(f"Specificity: {specificity}%")

    return tp, tn, fp, fn, accuracy, sensitivity, specificity

    # opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

image_name = "21"

train_src = cv2.imread('../../resources/Task2/Training/original_retinal_images/' + image_name + '_training.tif', cv2.IMREAD_COLOR)
train_mask = np.array(Image.open('../../resources/Task2/Training/background_masks/' + image_name + '_training_mask.gif'))
ground_truth = np.array(Image.open('../../resources/Task2/Training/blood_vessel_segmentation_masks/' + image_name + '_manual1.gif'))

# extract only green from image
train_green = train_src[:, :, 1]

# Pre processing

# perform histogram stretching
layer_pre = exposure.equalize_hist(train_green, mask=train_mask).astype(np.float32)

# perform bilateral filter
layer_pre = cv2.bilateralFilter(layer_pre, 4, 150, 150)


# create Gaussian pyramid for function input
train_pyramid, mask_pyramid = gaussian_pyramid(layer_pre, train_mask, NUM_LAYERS)

fig = plt.figure()
fig.suptitle("Layers")
out_pyramid = []

for i in range(NUM_LAYERS):
    layer = train_pyramid[i].astype(np.float32)
    mask = mask_pyramid[i]

    # perform hessian analysis
    layer_out = hessian_analysis(layer)

    # resize pyramid layers to original image size
    layer_out = resize_image(layer_out, train_green)

    layer_thresh = apply_morphological_postprocessing(layer_out, PARAMETERS[str(i)]["mode"])

    # apply hysteresis thresholding
    layer_thresh = hysteresis_thresholding(layer_thresh, PARAMETERS[str(i)]["h_min"], PARAMETERS[str(i)]["h_max"], 255)

    # apply masking
    layer_thresh = get_masked_output(layer_thresh, train_mask)

    fig.add_subplot(2, 3, i+1)
    plt.imshow(layer_out, cmap='gray')
    plt.axis('off')
    plt.title('Hessian analysis')

    fig.add_subplot(2, 3, i+4)
    plt.imshow(layer_thresh, cmap='gray')
    plt.axis('off')
    plt.title('Post proc.')

    print('Completed Layer %s' % i)
    out_pyramid.append(layer_thresh)

plt.show()

train_out = np.zeros(train_green.shape, dtype=np.float32)

# combine all images to create final image
for i in range(NUM_LAYERS):
    layer = out_pyramid[i]
    train_out = cv2.bitwise_or(src1=train_out, src2=layer)

cv2.imshow('Output', train_out)
train_out = apply_morphological_postprocessing(train_out, 3)
cv2.imshow('Morph Output', train_out)
print (train_out.shape)
print(ground_truth.shape)
get_metrics(train_out, ground_truth)

cv2.waitKey(0)