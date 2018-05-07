import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    print("Path removed")
else:
    pass
import cv2
import numpy as np
import pickle
import pandas as pd
import Get_Position
import blr
from math import atan2


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

        self.class_list = ["0","1","2","3","4","5",'6','7','8','9','zero' ,'one','two','three','four','five','six','seven','eight','nine','zero_TH' ,'one_TH','two_TH','three_TH','four_TH','five_TH','six_TH','seven_TH','eight_TH','nine_TH']

    def create_camera_instance(self, id=0):

        self.cap = cv2.VideoCapture(id)

    def release_camera_instance(self):

        self.cap.release()

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

    def one_time(self):
        model_name = "NNC.sav"
        with open(model_name, 'rb') as model_f:
            model = pickle.load(model_f) 
        # model = pickle.load(open(model_name, 'rb'))
        ret, frame = self.cap.read()
        self.img_size = frame.shape
        edges = cv2.Canny(frame,100,200)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        ret, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        selected_contour = []
        left_side = []
        right_side = []
        new_left_side = []
        new_right_side = []
        new_approx = []
        midpoint = []
        transform = None
        row = self.create_row(100, 729)
        real_contour =[]
        class_num =[]
        midpoint_list = []
        corner_list = []


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
                        predict_result = model.predict(feature.values.reshape(1,-1))
                        midpoint.append((cX, cY+50))
                        midpoint_list.append([cX, cY])
                        class_num.append(int(predict_result))
                        corner_list.append(approx)

                    left_side = []
                    right_side = []
                    new_left_side = []
                    new_right_side = []
                    new_approx = []
                    transform = None

        cv2.drawContours(frame, selected_contour, -1, (0,0,255), 3)
        for disp in range(len(midpoint)):
                    cv2.putText(frame,self.class_list[class_num[disp]], midpoint[disp], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3,cv2.LINE_8)
        

        cv2.imshow("Camera", frame)
        cv2.imshow("Edge", edges)
        cv2.waitKey(0)
        frame = None
        return class_num, midpoint_list, corner_list, []


    def real_time(self):
        model_name = "NNC.sav"
        # model_name = "Last_mlp.sav"
        # model_name = "Last_svc.sav"
        # model_name = "D:\\SVM\\svm_model_hist_FULL.sav"
        # model_name = "/media/phurinpat/Local Disk/SVM/svm_model_hist.sav"
        # model_name = "D:\\KNN\\FS_testKNN_50_uniform_2.sav"
        # model_name = 'C:\\Users\\PHURINPAT\\PycharmProjects\\Module 8-9\\RANDOM_FORREST\\FS_testRF_30_28_2.sav'
        model = pickle.load(open(model_name, 'rb'))
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
                if cv2.contourArea(cnt) >= 1500:
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
                            # time_ = time.time()
                            predict_result = model.predict(feature.values.reshape(1,-1))
                            # print("Time used: ",time.time()-time_)
                            midpoint.append((cX, cY+50))
                            str_class.append(self.class_list[int(predict_result)])
                            for disp in range(0, len(midpoint)):
                                cv2.putText(frame,str_class[disp], midpoint[disp], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3,cv2.LINE_8)
                        # else:
                        #     cv2.destroyWindow("Crop")
                        #     # cv2.destroyWindow("transform")
                        #     cv2.destroyWindow("get_Histogram")
                        #     # print(feature)

                        # break
                        left_side = []
                        right_side = []
                        new_left_side = []
                        new_right_side = []
                        new_approx = []
                        transform = None
            
            cv2.drawContours(frame, selected_contour, -1, (0,0,255), 2)
            
            for disp in range(0, len(midpoint)):
                cv2.putText(frame,str_class[disp], midpoint[disp], cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),3,cv2.LINE_8)

            cv2.imshow("Camera", frame)
            cv2.imshow("Edge", edges)
				

            if cv2.waitKey(1) == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break

class Obj_Sent:
    number_of_plate = 0
    class_list = []
    position = []
    corner = []
    def __init__(self, num_of_plate=0, class_num_plate=[], pos_plate=[], corner_=[]):
        self.number_of_plate = num_of_plate
        self.class_list = class_num_plate 
        self.position = pos_plate
        self.corner = corner_

def convert_pos(pos_list=[],mtx=[],homo=[],mode=1,inverse=False):
    convert = Get_Position.World()
    pos_list_convert = []
    sub_pos_list = []
    if mode == 1:
        if not inverse:
            if len(pos_list):
                for i,j in enumerate(pos_list,0):
                    x_y = convert.calculate_World_coor(j[0],j[1],mtx,homo)
                    x = float(x_y[1][0])
                    y = float(x_y[1][1])
                    pos_list_convert.append([x, y])

                return pos_list_convert
        else:
            if len(pos_list):
                for i,j in enumerate(pos_list,0):
                    x_y = convert.calculate_World_coor_reverse(j[0],j[1],mtx,homo)
                    x = float(x_y[1][0])
                    y = float(x_y[1][1])
                    pos_list_convert.append([x, y])

                return pos_list_convert
    else:
        if not inverse:
            if len(pos_list):
                for k in range(len(pos_list)):
                    for i,j in enumerate(pos_list[k],0):
                        x_y = convert.calculate_World_coor(j[0][0],j[0][1],mtx,homo)
                        x = float(x_y[1][0])
                        y = float(x_y[1][1])
                        sub_pos_list.append([x, y])
                        
                    pos_list_convert.append(sub_pos_list)
                    sub_pos_list = []

                return pos_list_convert
        else:
            if pos_list is not None:
                if len(pos_list):
                    for k in range(len(pos_list)):
                        for i,j in enumerate(pos_list[k],0):
                            # print(j)
                            x_y = convert.calculate_World_coor_reverse(j[0],j[1],mtx,homo)
                            x = float(x_y[1][0])
                            y = float(x_y[1][1])
                            sub_pos_list.append([x, y])
                            
                        pos_list_convert.append(sub_pos_list)
                        sub_pos_list = []

                    return pos_list_convert

def find_world_coor(blr='l',xy=[]):
    if blr == 'l':
        return [xy[0], 473, xy[1]]
    elif blr == 'r':
        return [-xy[0], -473, xy[1]]
    elif blr == 'br' or blr == 'bl' or blr == 'b':
        return [xy[1], -xy[0], 27]

def find_distance(p1=[], p2=[]):
    vector = []
    for i in range(3):
        vector.append(p1[i]-p2[i])
    distance = np.linalg.norm(vector)
    return distance

def get_rpy(cor_l=[], scene='l'):
    pi = np.pi
    print(cor_l, "coor")
    coor = sorted(cor_l, key=lambda k: [k[1], k[0]])
    bot_side =[]
    top_side = []
    bot_side.append(coor[0])
    bot_side.append(coor[1])
    top_side.append(coor[2])
    top_side.append(coor[3])
    new_bot_side = sorted(bot_side, key=lambda k: [k[0], k[1]])
    new_top_side = sorted(top_side, key=lambda k: [k[0], k[1]])

    angle = atan2(new_top_side[1][1]-new_top_side[0][1], new_top_side[1][0]-new_top_side[0][0])
    print(np.degrees(angle), 'Rad')
    if scene == 'l':
        return [pi/2, angle-(pi/2), pi]
    elif scene == 'r':
        return [pi/2, angle-(pi/2), 0]
    elif scene == 'bl' or scene == 'br' or scene == 'b':
        return[pi, 0 ,angle]

def pack_data(ca_l=[], pos_l=[], cor_l=[], rw_l=[], dat_pack=[], scene='l'):
    if rw_l:
        for counter in range(len(rw_l)):
            rw_l[counter][0] = rw_l[counter][0] * -1
            rw_l[counter][1] = rw_l[counter][1] * -1
    for counter, i in enumerate(pos_l):
        usable = True
        realworld = [j*1000 for j in rw_l[counter]]
        realworld = find_world_coor(blr=scene, xy=realworld)
        if len(dat_pack):
            for pack in dat_pack:
                distance =find_distance(pack[0], realworld)
                print(distance, counter)
                if distance <= 105:
                    usable = False
                    print('Unusable: Duplicate card.')
                    break
            if usable:
                dat_pack.append([realworld, get_rpy(cor_l[counter],scene), ca_l[counter]])
        else:
            dat_pack.append([realworld, get_rpy(cor_l[counter],scene), ca_l[counter]])

    return dat_pack



if __name__ == "__main__":
    data_pack = []
    convert = Get_Position.World()
    newcammtx = Get_Position.newcammtx
    # homo = Get_Position.homo
    
    rtp = Real_Time_Predict()
    rtp.create_camera_instance(0)
    rtp.create_HogDescriptor()

    rtp.release_camera_instance()
    rtp.create_camera_instance(0)

    #open camera and read model and predict get centroid of picture and 4 points of conner
    cardlist, midpointlist, cornerlist, realworldlist = rtp.one_time()

    #get_homo, put pantile's list by pan is q1 and tilt is q2 , blr is scene ('l', 'r', 'blbr')
    homo = blr.get_homo(q1_=-90,q2_=0,blr_='l')[1]

    # realworldlist is the function that convert centroid of picture to centroid of picture with respect to world coordinate
    realworldlist = convert_pos(midpointlist,newcammtx,homo)

    # same as realworldlist but it using 4 points of conner
    cornerworldlist = convert_pos(cornerlist,newcammtx,homo,mode=0,inverse=False)

    # data_pack include 1.cardlist is class og each card, 2.midpointlist is useless, 3.cornerworldlist is cornerworldlist
    #                   4.realworldlist is realworldlist, 5.data_pack (it will send to MATLAB later) 6.parameter that tell scene
    data_pack = pack_data(cardlist, midpointlist, cornerworldlist, realworldlist, data_pack, 'l')
    print(data_pack)
    cornerworldlist = None
    
    # ***************************************************************************************************************************************************************************
    cv2.waitKey(0)
    rtp.release_camera_instance()
    rtp.create_camera_instance(0)
    cardlist, midpointlist, cornerlist, realworldlist = rtp.one_time()
    homo = blr.get_homo(q1_=90,q2_=0,blr_='r')[2]
    realworldlist = convert_pos(midpointlist,newcammtx,homo)
    cornerworldlist = convert_pos(cornerlist,newcammtx,homo,mode=0,inverse=False)
    data_pack = pack_data(cardlist, midpointlist, cornerworldlist, realworldlist, data_pack, 'r')
    print(data_pack)

    # ***************************************************************************************************************************************************************************
    # print(convert_pos(obj[2],newcammtx,homo,mode=0))
    # print(obj[2][0])

    # print('Number of plate: \n\t{}\nClass list: \n\t{}\nPosition: \n\t{}\nCorner: \n\t{}\n'.format(result.number_of_plate, result.class_list, result.position, result.corner))
    # print(result.position)
    # print(result.number_of_plate)
    # print(result.class_list)
    # print(result.corner)


