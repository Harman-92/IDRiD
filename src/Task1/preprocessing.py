import cv2

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
