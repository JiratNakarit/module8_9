import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    print("Path removed")
else:
    pass
import cv2
import numpy as np
import pickle
import copy
import pandas as pd
import sklearn
import time


class Real_Time_Predict:

    def __init__(self):
        self.IMAGE_SIZE = 50
        self.IMAGE_SIZE_ORI = 80

        self.winSize = (self.IMAGE_SIZE, self.IMAGE_SIZE)
        self.blockSize = (10, 10)
        self.blockStride = (5, 5)
        self.cellSize = (10, 10)
        self.nbins = 9
        self.derivAperture = 1
        self.winSigma = 1.0
        self.histogramNormType = 0
        self.L2HysThreshold = 0.2
        self.gammaCorrection = 1
        self.nlevels = 64

        self.class_list = ["0","1","2","3","4","5",'6','7','8','9','zero' ,'one','two','thee','four','five','six','seven','eight','nine','zero_TH' ,'one_TH','two_TH','thee_TH','four_TH','five_TH','six_TH','seven_TH','eight_TH','nine_TH']

    def create_camera_instance(self, id=0):

        self.cap = cv2.VideoCapture(id)

    def crop_(self, plate_img):
        top_bot = []
        left_right = []
        kernel = np.ones((3, 3), np.uint8)
        img = plate_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(img,175,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(img,70,200)

        # invt = cv2.bitwise_not(copy.copy(thresh))
        invt = cv2.dilate(edges, kernel, iterations=1)

        ret, contours, _ = cv2.findContours(invt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i in contours:
            x, y, w, h = cv2.boundingRect(i)
            top_bot.append(y)
            top_bot.append(y + h)
            left_right.append(x)
            left_right.append(x + w)

        if len(top_bot) != 0:
            word_crop = img[min(top_bot):max(top_bot), min(left_right):max(left_right)].copy()
            word_crop = cv2.resize(word_crop, (50, 50))

        # cv2.imshow("img", img_)
        # cv2.imshow("invt", invt)
        # cv2.imshow("Crop", word_crop)
        # cv2.waitKey(1)
        # cv2.destroyAllWindows()
            return word_crop
        return None

    def create_HogDescriptor(self):
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins,
                                     self.derivAperture, self.winSigma, self.histogramNormType, self.L2HysThreshold,
                                     self.gammaCorrection, self.nlevels)

    def get_HOG(self, img_):

        hog_ = self.hog.compute(img_).reshape(-1)
        return hog_

    def get_Histogram(self, img_):
        row, col = np.zeros(self.IMAGE_SIZE), np.zeros(self.IMAGE_SIZE)

        # _, bw = cv2.threshold(img_, 100, 255, cv2.THRESH_BINARY)
        bw = cv2.adaptiveThreshold(img_,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,5)
        cv2.imshow("get_Histogram",bw)
        for i in range(0, self.IMAGE_SIZE):
            for j in range(0, self.IMAGE_SIZE):
                if bw[i][j] == 0:
                    row[i] += 1
                    col[j] += 1
        hist = np.hstack((row, col))
        row, col = np.zeros(self.IMAGE_SIZE), np.zeros(self.IMAGE_SIZE)
        return hist

    def pack_feature(self, hist, hog):
        hist /= 50
        feature = np.hstack((hist, hog))
        # feature = np.hstack((np.nan, feature))
        return feature

    def create_row(self, hist, hog):
        colname = []

        for i in range(1, hist + 1):
            colname.append('hist' + str(i))
        for j in range(1, hog + 1):
            colname.append('hog' + str(j))

        return colname


    def camera(self):
        model_name = "NNC.sav"
        # model_name = "D:\\SVM\\svm_model_hist_FULL.sav"
        # model_name = "/media/phurinpat/Local Disk/SVM/svm_model_hist.sav"
        # model_name = "D:\\KNN\\FS_testKNN_50_uniform_2.sav"
        # model_name = 'C:\\Users\\PHURINPAT\\PycharmProjects\\Module 8-9\\RANDOM_FORREST\\FS_testRF_30_28_2.sav'
        clf = pickle.load(open(model_name, 'rb'))
        # list_ = np.load('FS.npy')
        while(1):
            ret, frame = self.cap.read()
            edges = cv2.Canny(frame,100,200)
            ret, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            selected_contour = []
            left_side = []
            right_side = []
            new_left_side = []
            new_right_side = []
            new_approx = []
            midpoint = []
            str_class = []
            transform = None
            row = self.create_row(100, 729)    
            # with open(model_name, 'rb') as f:
            #     clf = pickle.load(f, encoding='latin1')

            for cnt in contours:
                if cv2.contourArea(cnt) >= 3000:
                    epsilon = 0.1*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    if len(approx) == 4:
                        selected_contour.append(cnt)
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

                        transform = cv2.getPerspectiveTransform(pts1, pts2)
                        
                        crop = cv2.warpPerspective(frame, transform, (300, 300))
                        crop = crop[30:270, 30:270]
                        crop = cv2.resize(crop,(80, 80))

                        cv2.imshow("transform", crop)

                        word_crop = self.crop_(crop)

                        if word_crop is not None:

                            M = cv2.moments(cnt)
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])


                            cv2.imshow("Crop", word_crop)
                            hog_feature = self.get_HOG(img_=word_crop)
                            hist_feature = self.get_Histogram(img_=word_crop)
                            feature = pd.Series(self.pack_feature(hist_feature, hog_feature), index=row)
                            # fs_feature = pd.DataFrame(feature.reshape(1,-1), columns=row)
                            # for fs in list(fs_feature):
                            #     if fs not in list_:
                            #         fs_feature.drop(fs, inplace=True, axis=1)
                            time_ = time.time()
                            predict_result = clf.predict(feature.values.reshape(1,-1))
                            print("Time used: ",time.time()-time_)
                            midpoint.append((cX, cY+50))
                            str_class.append(self.class_list[int(predict_result)])
                            for disp in range(0, len(midpoint)):
                                cv2.putText(frame,str_class[disp], midpoint[disp], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3,cv2.LINE_8)
                            # print(feature)

                        # break
                        left_side = []
                        right_side = []
                        new_left_side = []
                        new_right_side = []
                        new_approx = []
                        transform = None

            cv2.drawContours(frame, selected_contour, -1, (0,0,255), 1)

            cv2.imshow("Camera", frame)
            cv2.imshow("Edge", edges)
				

            if cv2.waitKey(1) == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break


rtp = Real_Time_Predict()
rtp.create_camera_instance(0)
rtp.create_HogDescriptor()
rtp.camera()