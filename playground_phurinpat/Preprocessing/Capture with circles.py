import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

# import time

quantity = 109
type_plate_list = ['N', 'E', 'T']
row_img = np.zeros([100, 100])
counter = 0
num_circle = 0
list_circle = []


def init():
    global num_plate, new_contour, new_approx, left_side, right_side, contour_size, row_img
    num_plate = 0
    new_contour = []
    new_approx = []
    left_side = []
    right_side = []
    contour_size = []


camera = cv2.VideoCapture(0)


def main(plate_num, plate_type, fontnum):
    global counter, quantity, type_plate_list, row_img, num_circle
    init()
    # time.sleep(2.5)
    while camera.isOpened():
        # img = cv2.imread('plate04.png')
        ret, img = camera.read()
        img_show = img.copy()
        edges = cv2.Canny(img, 100, 200)
        im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            M = cv2.moments(i)
            # if 60000 <= M['m00'] < 120000:
            if 1 <= M['m00'] < 40000:
                new_contour.append(i)
                contour_size.append(M['m00'])
        biggest = contour_size.index(max(contour_size))
        print(contour_size[biggest])
        print('Index of Biggest contour:', biggest)

        print("Number of New contours:", len(new_contour))

        cv2.drawContours(img, [new_contour[biggest]], 0, (0, 255, 0), 5)

        approx = cv2.approxPolyDP(new_contour[biggest], 0.01 * cv2.arcLength(new_contour[biggest], True), True)
        for i in approx:
            new_approx.append(i[0])
        old_coor = sorted(new_approx, key=lambda k: [k[0], k[1]])
        left_side.append(old_coor[0])
        left_side.append(old_coor[1])
        right_side.append(old_coor[2])
        right_side.append(old_coor[3])
        new_left_side = sorted(left_side, key=lambda k: [k[1], k[0]])
        new_right_side = sorted(right_side, key=lambda k: [k[1], k[0]])

        pts1 = np.float32([new_left_side[0], new_left_side[1], new_right_side[0], new_right_side[1]])
        pts2 = np.float32([[0, 0], [0, 300], [300, 00], [300, 300]])

        L = cv2.getPerspectiveTransform(pts1, pts2)

        crop = cv2.warpPerspective(img_show, L, (300, 300))
        copy_crop = crop.copy()

        gray_crop = crop.copy()
        gray_crop = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)
        circle = cv2.HoughCircles(gray_crop, cv2.HOUGH_GRADIENT, 2, 10, param1=30, param2=50, minRadius=0, maxRadius=50)

        if circle is not None:
            circle = np.uint16(np.around(circle))

            for i in circle[0, :]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        resize_image = cv2.resize(copy_crop, (100, 100))
        resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)
        ret, bw = cv2.threshold(resize_image, 127, 255, cv2.THRESH_BINARY)

        if counter in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            if circle is not None:
                print('Number of Circle:', len(circle[0]))
                num_circle = len(circle[0])
            else:
                num_circle = 0

            if counter == 9:
                list_circle.append(num_circle)

            row_img = resize_image
        else:
            if circle is not None:
                print('Number of Circle:', len(circle[0]))
                num_circle = len(circle[0])
            else:
                num_circle = 0
            row_img = np.hstack([row_img, resize_image])
            list_circle.append(num_circle)

        init()
        counter += 1
        cv2.imshow("camera", img)
        cv2.waitKey(1)
        if counter == quantity:
            break

    cv2.imshow('Row', row_img)
    cv2.imwrite(str(plate_num) + type_plate_list[plate_type] + str(fontnum) + '.png', row_img)
    with open('circle' + str(plate_num) + type_plate_list[plate_type] + str(fontnum) + '.csv', 'w') as csvfile:
        file = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i, j in enumerate(list_circle):
            file.writerow(str(j))

    # cv2.waitKey(0)
    cv2.destroyAllWindows()


main(plate_num=9, plate_type=2, fontnum=5)

'''
img[:10,:10] = sample[i]
'''
