# Import module ________________________________________________________________________________________________________
import time
import math
import ast

# Import class _________________________________________________________________________________________________________
from Dynamixel import *
from Bord import *
from RealtimePredict import *
from blr import *
from Get_Position import *
from Planning import PathPlan

class Main:

    class Const:
        # Const. Position
        Home = [180,90,30,90,60,90]
        Front = [0,80,30,90,60,90]
        Right = [90,80,30,90,60,90]

        # Variable _____________________________________________________________________________________________________
        Camera = [[-50,-70],
                   [90,-30],[90,-10],[90,10],[90,25],
                   [75,22],[70,10],[70,0],[70,-10],[70,-20],
                   [60,-20],[60,-10],[70,0],[70,10],[70,22]]


    def __init__(self,nomodeset=0):
        # Call class ___________________________________________________________________________________________________
        if nomodeset == 1:
            self.KHONG = Board('COM4',115200)
        elif nomodeset == 2:
            self.CAMER = Dynamixel('COM3',1000000)
        elif nomodeset == 0:
            self.KHONG = Board('COM4',115200)
            self.CAMER = Dynamixel('COM3',1000000)
        else:
            raise NameError('NoMode not found (your mode is ',nomodeset)

        # homo = Get_Position.homo
        self.rtp = Real_Time_Predict()
        self.rtp.create_camera_instance(0)
        self.rtp.create_HogDescriptor()

        self.convert = Get_Position.World()
        self.newcammtx = Get_Position.newcammtx


    # Function _________________________________________________________________________________________________________
    def camera(self,pantilt):
        self.CAMER.PAN(pantilt[0])
        self.CAMER.TILT(pantilt[1])
        self.CAMER.WaitFinish([1,9])

    def inv_pantilt(self,pantilt):
        result = pantilt[::-1].copy()
        for i in range(len(result)):
            result[i][0] = result[i][0]*-1
        return result[:7]

    def cmKhong(self,position):
        self.KHONG.SetPosition(position)
        time.sleep(0.1)
        #KHONG.WaitFinish()

    def cam_clf(self,DATA_PACK,brl,pt):
        self.rtp.release_camera_instance()
        self.rtp.create_camera_instance(0)

        #open camera and read model and predict get centroid of picture and 4 points of conner
        cardlist, midpointlist, cornerlist, realworldlist = self.rtp.one_time()

        #get_homo, put pantile's list by pan is q1 and tilt is q2 , blr is scene ('l', 'r', 'blbr')
        homo = blr.get_homo(q1_=pt[0], q2_=pt[1], blr_=brl)[1]

        # realworldlist is the function that convert centroid of picture to centroid of picture with respect to world coordinate
        realworldlist = convert_pos(midpointlist, newcammtx, homo)

        # same as realworldlist but it using 4 points of conner
        cornerworldlist = convert_pos(cornerlist, newcammtx, homo, mode=0, inverse=False)

        # data_pack include 1.cardlist is class og each card, 2.midpointlist is useless, 3.cornerworldlist is cornerworldlist
        #                   4.realworldlist is realworldlist, 5.data_pack (it will send to MATLAB later) 6.parameter that tell scene
        data_pack = pack_data(cardlist, midpointlist, cornerworldlist, realworldlist, DATA_PACK, brl)
        return data_pack

    # Loop camera(pan,tilt),classify, position _________________________________________________________________________
    def Step1FindCard(self):
        T_CLF = time.time()
        STATE = 1
        CARD_POSITION = []

        for PT in Main.Const.Camera:
            if STATE == 1:
                # 1. Rotate J1 90 degree
                self.cmKhong(Main.Const.Right)
                # 2. Pan CAM (-50,-70)
                self.camera(PT)
                # 3. Predict & Position
                CARD_POSITION = self.cam_clf(CARD_POSITION,'br',PT)
            else:
                # 4. Rotate J1 90 degree
                if STATE == 2:
                    self.cmKhong(Main.Const.Front)
                # 5. Pan CAM [95,-30],[95,-10],[95,15],[95,22],[70,22],[70,-10],[70,-30]
                self.camera(PT)
                # 6. Predict & Position
                CARD_POSITION = self.cam_clf(CARD_POSITION,'r',PT)
            STATE += 1

        for PT in INV_PANTILT:
            if STATE != 16:
                # 7. Pan CAM [-95,-30],[-95,-10],[-95,15],[-95,22],[-70,22],[-70,-10],[-70,-30]
                self.camera(PT)
                # 8. Predict & Position
                CARD_POSITION = self.cam_clf(CARD_POSITION,'l',PT)
            else:
################# Can't rotate Joint 1 to -90 degree ###################################################################
                # 9. Rotate J1 90 degree
                #self.cmKhong()
                # 10. Pan CAM
                #self.camera(PT)
                # 11. Predict & Position
                #CARD_POSITION = self.cam_clf(CARD_POSITION,'bl',PT)
                pass

        # Set home position
        self.CAMER.PANTILT(0)
        self.cmKhong(Main.Const.Home)
        ENDT_CLF = T_CLF - time.time()
        return CARD_POSITION, ENDT_CLF

# Planning (MATLAB) ____________________________________________________________________________________________________
    def Step2PathPlan(self,CardPosition):
        T_PLN = time.time()
        CHIN = PathPlan(CardPosition)
        PATH = CHIN.EvaluateTraject()
        ENDT_PLN = time.time() - T_PLN
        return PATH,ENDT_PLN


# Command Khong ________________________________________________________________________________________________________
    def Step3CommandKhong(self,PATH):
        T_LLV = time.time()
        #Traject
        pose = [180,90,30,90,90,60]
        old_pose = [90,60,90]
        for path in range(len(PATH)):
            #Sub Traject
            for STpath in range(len(PATH[path])):
                via = PATH[path][STpath][len(PATH[path][STpath])-1]
                for k in range(len(PATH[path][STpath])):
                    pose = [0,0,0,0,0,0]
                    pose[3:6] = old_pose
                    tj = PATH[path][STpath][k]
                    if k != len(PATH[path][STpath])-1:
                        pose[0:3] = tj[0:3]
                    else:
                        pose = via
                    old_pose = pose[3:6]
                    self.cmKhong(pose)
                    print('via: ',pose)
            if (path+1)%2 ==0:
                if (path+1)%4 == 0:
                    self.KHONG.SetGrip(0)
                else:
                    self.KHONG.SetGrip(1)

            print('Finish SUB ',path)
            time.sleep(2)

        self.KHONG.SetGrip(0)
        ENDT_LLV = time.time() - T_LLV
        return ENDT_LLV

if __name__ == '__main__':
    Sequen = Main(nomodeset=0)
    CardPosition, T_1 = Sequen.Step1FindCard()
    Path, T_2 = Sequen.Step2PathPlan(CardPosition)
    T_3 = Sequen.Step3CommandKhong(Path)

# In Conclusion ________________________________________________________________________________________________________
    print('Summary')
    print('1. Time for find card: ',T_1)
    print('2. Time for Planning : ',T_2)
    print('3. Time for Running  : ',T_3)

'''
pi = math.pi

CARD_POSITION = [[[150.,275.,27.],[pi,0.,0.],0.],
                [[150.,-275.,27.],[pi,0.,0.],19.],
                [[500.,473.,800.],[pi/2,-pi/2,pi],21.],
                [[380.,473.,600.],[pi/2,-pi/2,pi],8.],
                [[600.,473.,300.],[pi/2,-pi/2,pi],1.],
                [[300.,473.,825.],[pi/2,-pi/2,pi],5.],
                [[300.,-473.,200.],[pi/2,-pi/2,0.],17.],
                [[450.,-473.,500.],[pi/2,-pi/2,0.],29.],
                [[620.,-473.,200.],[pi/2,-pi/2,0.],20.],
                [[620.,-473.,800.],[pi/2,-pi/2,0.],13.]]
'''
