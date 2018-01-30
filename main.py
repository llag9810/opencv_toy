import cv2
import transform
import numpy as np
from matplotlib import pyplot as plt

src = cv2.imread('1.jpg')
src = cv2.resize(src, (640, 960))
src = transform.transform(src)
src_copy = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.GaussianBlur(gray, (5, 5), 0)

x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

sobel = cv2.addWeighted(absX, 0.6, absY, 0.6, 0)

blurred = cv2.blur(sobel, (3, 3))

ret, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)
bin_copy = binary.copy()
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
_, contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


rect_list = []
for i in range(len(contours)):
    x, y, l, w = cv2.boundingRect(contours[i])
    rect_list.append((x, y, l, w))
    # cv2.rectangle(sobel, (x, y), (x + l, y + w), (255, 255, 0), 5)


def is_button(rect):
    x, y, l, w = rect
    return 70 < abs(l) < 150 and 70 < abs(w) < 150


key_list = filter(is_button, rect_list)


def sort_key_list(x, y):
    x1, y1, _, _ = x
    x2, y2, _, _ = y
    if x1 - x2 > 30:
        return 1
    elif x1 - x2 < -30:
        return -1
    elif y1 - y2 > 30:
        return 1
    elif y1 - y2 < -30:
        return -1
    else:
        return 0


key_list.sort(sort_key_list)

for i in range(len(key_list)):
    x, y, l, w = key_list[i]
    cv2.rectangle(src, (x, y), (x + l, y + w), (255, 255, 0), 5)

print key_list
res = binary

def get_key_image(rect):
    x, y, l, w = rect
    return gray[x+5:x+l-15, y+10:y+w-10]


key_img = map(get_key_image, key_list)
key_copy = list(key_img)

for i in range(len(key_img)):
    ret, key_img[i] = cv2.threshold(key_img[i], 55, 255, cv2.THRESH_BINARY)

sample_src = cv2.imread('sample2.png', 0)
sample_src = cv2.resize(sample_src, (500, 500))
blurred = cv2.blur(sample_src, (3, 3))
ret, sample_src = cv2.threshold(sample_src, 127, 255, cv2.THRESH_BINARY_INV)
_, sample_contours, _ = cv2.findContours(sample_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sample_rect_list = []
for i in range(len(contours)):
    x, y, l, w = cv2.boundingRect(contours[i])
    sample_rect_list.append((x, y, l, w))
    # cv2.rectangle(sample_src, (x, y), (x + l, y + w), (255, 255, 0), 5)

sample_key_list = filter(is_button, sample_rect_list)
sample_key_list.sort(sort_key_list)


def get_sample_key_image(rect):
    x, y, l, w = rect
    return sample_src[x:x+l, y:y+w]


sample_key_img = map(get_sample_key_image, sample_key_list)


def get_black_pixel_proprotion(img):
    length, width = img.shape
    black = 0
    for i in range(length):
        for j in range(width):
            if img[i][j] < np.uint8(25):
                black += 1
    return float(black) / (length * width)


res_sample = map(get_black_pixel_proprotion, sample_key_img)
res_real = map(get_black_pixel_proprotion, key_img)

result = []
for i in range(len(res_real)):
    result.append(res_real[i] / res_sample[i])

max_res = max(result)
min_res = min(result)

res = map(lambda a: round(0.937 * round((a-min_res) / (max_res - min_res), 3), 3), result)

print res

for i in range(len(key_img)):
    plt.subplot(4, 4, i+1), plt.imshow(key_copy[i], 'gray')
    plt.title(res[i])
    plt.xticks([]), plt.yticks([])
plt.show()

cv2.destroyAllWindows()
