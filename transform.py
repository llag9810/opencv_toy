import cv2
import numpy as np


def transform(img):
    img = cv2.resize(img, (640, 960))
    img_copy = img.copy()

    pts = []

    def get_point(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(img_copy, (x, y), 6, (255, 0, 0), 3)
            pts.append([x, y])

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', get_point)

    while len(pts) < 4:
        cv2.imshow('img', img_copy)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyWindow('img')
    print(pts)

    pts = np.float32(pts)

    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    M = cv2.getPerspectiveTransform(pts, pts2)
    dst = cv2.warpPerspective(img, M, (500, 500))

    return dst


if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    res = transform(img)

    while True:
        cv2.imshow('img', res)
        if cv2.waitKey(0) == 27:
            break