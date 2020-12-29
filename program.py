from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path('./data')
IMAGES_DIR = DATA_DIR / 'images'
LABELS_DIR = DATA_DIR / 'labels'
RESULTS_DIR = DATA_DIR / 'results'


def largest_blob(mask):
    # firstly find all contours and then the largest
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)

    # create a mask with the largest contour
    largest_contour_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, cv2.FILLED)

    # apply union to the original and created masks
    return cv2.bitwise_and(mask, largest_contour_mask)


def add_padding(mask, k=1, c=0):
    h, w = mask.shape

    bigger_img = np.full((h + 2 * k, w + 2 * k), c, dtype=np.uint8)
    bigger_img[k:(h + k), k:(w + k)] = mask

    return bigger_img


def remove_padding(mask, k=1):
    h, w = mask.shape
    h, w = h - 2 * k, w - 2 * k  # calculate original values
    return mask[k:(h + k), k:(w + k)]


def fill_holes(mask, fill=255, k=1):
    # adds padding
    flood_fill_mask = add_padding(mask, k=k)

    # applies flood fill algorithm
    h, w = flood_fill_mask.shape
    mask_mask = np.zeros((h + 2 * k, w + 2 * k), np.uint8)
    cv2.floodFill(flood_fill_mask, mask_mask, (0, 0), fill)

    # removes padding
    flood_fill_mask = remove_padding(flood_fill_mask, k=k)

    # inverts the mask
    flood_fill_mask_inv = cv2.bitwise_not(flood_fill_mask)

    # returns union over original mask and the calculated one
    return mask | flood_fill_mask_inv


def process(img):
    """
    The main function of this program. It that applies creates fingernails mask for one input image.

    :param img: input image
    :return: fingernails mask
    """

    # convert image to HSV color space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # filter skin color
    mask = cv2.inRange(img_hsv, (0, 50, 80), (40, 255, 255))

    mask = cv2.blur(mask, (3, 3))

    # find largest object that represents hands
    mask = largest_blob(mask)

    # the same number of iterations and kernel should be used for dilation and erosion
    iterations = 25
    kernel = np.ones((7, 7), np.uint8)

    # cover more fingernail areas and fill the holes in the object
    mask = cv2.dilate(mask, kernel, iterations=iterations)
    mask = fill_holes(mask)
    mask = cv2.erode(mask, kernel, iterations=iterations)

    # apply the mask to the original image
    hand = cv2.bitwise_and(img, img, mask=mask)

    # filter fingernail colors
    hand_hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hand_hsv, (90, 10, 115), (200, 132, 255))
    mask = cv2.bitwise_or(mask, cv2.inRange(hand_hsv, (0, 10, 100), (5, 121, 255)))

    # removing noise
    mask = cv2.medianBlur(mask, 9)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    return mask


def iou(label, image):
    intersection = np.logical_and(label, image)
    union = np.logical_or(label, image)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def images():
    """
    Generator that yields original image, label and the name of each image in IMAGES_DIR directory.
    :return:
    """
    for image_path in IMAGES_DIR.iterdir():
        label_path = LABELS_DIR / image_path.name

        if not label_path.exists():
            raise Exception(f'No label for image: {image_path}')

        image = cv2.imread(str(image_path))
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        yield image, label, image_path.name


def compare_results():
    """
    Goes through every image and for every one
    displays the original and with the original with the calculated mask applied.
    """
    for image, label, _ in images():
        cv2.imshow('org', image)
        mask = process(image)
        processed = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow('processed', processed)
        cv2.waitKey(0)


def get_score():
    """
    Goes through every image and for every one calculates IoU metric.
    Finally prints the average of all of them.
    """
    i, iou_sum = 0, 0
    for image, label, name in images():
        processed_mask = process(image)
        iou_score = iou(label, processed_mask)
        # print(f'{name}: {int(iou_score * 100)}%')
        iou_sum += iou_score
        i += 1
    iou_avg = iou_sum / i
    print('IoU avg:', iou_avg)


def save_results():
    """
    Goes through every image and for every one
    saves the calculated mask to the RESULTS_DIR directory.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    for image, label, name in images():
        processed_mask = process(image)
        cv2.imwrite(str(RESULTS_DIR/name), processed_mask)


if __name__ == '__main__':
    # compare_results()
    get_score()
    # save_results()
