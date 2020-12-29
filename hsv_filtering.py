import cv2
import numpy as np

from segmentation import *


def process1(img):
    mask = hand_mask(img)
    hand = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    mask = hsv_filter_mask(hsv, np.array([90, 10, 115]), np.array([200, 132, 255]))
    mask = mask | hsv_filter_mask(hsv, np.array([0, 10, 100]), np.array([5, 121, 255]))

    mask = blur(mask)
    mask = noise_removal(mask, it=3)
    return mask


if __name__ == '__main__':
    # compare_results(process_fn=process1)
    get_score(process_fn=process1)
