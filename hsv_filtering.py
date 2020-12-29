from segmentation import *


def process1(img):
    mask = hand_mask(img)
    hand = cv2.bitwise_and(img, img, mask=mask)

    hsv = cv2.cvtColor(hand, cv2.COLOR_BGR2HSV)
    mask = hsv_filter_mask(hsv, (90, 10, 115), (200, 132, 255))
    mask = mask | hsv_filter_mask(hsv, (0, 10, 100), (5, 121, 255))

    mask = blur(mask)
    mask = noise_removal(mask, it=3)
    return mask


if __name__ == '__main__':
    # compare_results(process_fn=process1)
    get_score(process_fn=process1)
