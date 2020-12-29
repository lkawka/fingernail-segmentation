import numpy as np
import cv2
from pathlib import Path
from matplotlib import pyplot as plt

DATA_DIR = Path('./data')
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'


def dice(pred, true, k=1):
    intersection = np.sum(pred[true == k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice


def copy_image(func):
    def inner(img, **kwargs):
        img = img.copy()
        return func(img, **kwargs)

    return inner


@copy_image
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


@copy_image
def color_channels(image):
    b = image.copy()
    # set green and red channels to 0
    b[:, :, 1] = 0
    b[:, :, 2] = 0

    g = image.copy()
    # set blue and red channels to 0
    g[:, :, 0] = 0
    g[:, :, 2] = 0

    r = image.copy()
    # set blue and green channels to 0
    r[:, :, 0] = 0
    r[:, :, 1] = 0
    cv2.imshow('r', r)
    cv2.imshow('g', g)
    cv2.imshow('b', b)


@copy_image
def contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


@copy_image
def canny(image):
    image = cv2.Canny(image, 50, 100)
    cv2.imshow('image', image)


@copy_image
def skin_segmentation(img):
    hsvim = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    skinRegionHSV = cv2.inRange(hsvim, lower, upper)
    blurred = cv2.blur(skinRegionHSV, (2, 2))
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    return thresh


@copy_image
def biggest_blob(img):
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output
    out = np.zeros(img.shape, np.uint8)
    cv2.drawContours(out, [cnt], -1, 255, cv2.FILLED)
    return cv2.bitwise_and(img, out)


@copy_image
def blur(img):
    return cv2.medianBlur(img, 9)


@copy_image
def noise_removal(image):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)


@copy_image
def noise_removal2(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


@copy_image
def add_padding(image, k=1, c=0):
    h, w = image.shape
    bigger = np.full((h + 2 * k, w + 2 * k), c, dtype=np.uint8)
    bigger[k:(h + k), k:(w + k)] = image
    return bigger


@copy_image
def remove_padding(img, k=1):
    h, w = img.shape
    h, w = h - 2 * k, w - 2 * k
    return img[k:(h + k), k:(w + k)]


@copy_image
def fill_holes(im_th):
    im_floodfill = add_padding(im_th)
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill = remove_padding(im_floodfill)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    return im_th | im_floodfill_inv


@copy_image
def make_bigger(img, k_shape=(5, 5), iterations=5):
    kernel = np.ones(k_shape, np.uint8)
    return cv2.dilate(img, kernel, iterations=iterations)


@copy_image
def make_smaller(img, k_shape=(5, 5), iterations=5):
    kernel = np.ones(k_shape, np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


@copy_image
def hand_mask(image):
    s = skin_segmentation(image)
    s = noise_removal(s)
    s = biggest_blob(s)
    s = blur(s)
    k_shape = (5, 5)
    iterations = 5
    s = make_bigger(s, k_shape=k_shape, iterations=iterations)
    s = fill_holes(s)
    s = make_smaller(s, k_shape=k_shape, iterations=iterations)
    return s


@copy_image
def kmeans(img, K=2):
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def iou(label, image):
    intersection = np.logical_and(label, image)
    union = np.logical_or(label, image)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


@copy_image
def process(image):
    h_mask = hand_mask(image)
    h = cv2.bitwise_and(image, image, mask=h_mask)

    s = h
    # p = cv2.normalize(s, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    # p = apply_brightness_contrast(p, contrast=64)
    # cv2.imshow('c', p)
    s = skin_segmentation(s)
    s = cv2.subtract(h_mask, s)
    return s


def images():
    for image_path in IMAGES_DIR.iterdir():
        label_path = LABELS_DIR / image_path.name

        if not label_path.exists():
            raise Exception(f'No label for image: {image_path}')

        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        yield image, label


def compare_results():
    for image, label in images():
        cv2.imshow('org', image)
        mask = process(image)
        processed = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('processed', processed)
        cv2.waitKey(0)


def get_score():
    i, sum = 0, 0
    for image, label in images():
        processed_mask = process(image)
        sum += iou(label, processed_mask)
        i += 1
    result = sum / i
    print('Result:', result)


if __name__ == '__main__':
    get_score()
    # compare_results()
