import cv2
import utils
import os

def reduce_resolution(image, size):
    dim = (size, size)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    print('Re-sized Dimensions : ', resized_image.shape)
    return resized_image


def scaleradius(img, scale):
    x = img[int(img.shape[0] / 2), :, :].sum(1)
    r = (x > x.mean() / 10).sum() / 2
    s = scale * 1.0 / r
    return cv2.resize(img, (0, 0), fx=s, fy=s)


def normalization(image):
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 100), -4, 128)
    return image


def enhance(image, clip_limit=3):
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


image = cv2.imread(os.path.join(utils.get_train_dir(), "images", "IDRiD_02.tif"))
print(type(image.shape))
# path=os.path.join()
# images = utils.read_images_from_folder(args.model_category, args.directory, args.image_format)
# print(len(images))

image = scaleradius(image, 300)
image = normalization(image)
print(image.shape)
image = enhance(image)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
