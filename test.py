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
#from Planning import PathPlan

# Call class ___________________________________________________________________________________________________________
#CAMER = Dynamixel('COM3',1000000)
KHONG = Board('COM4',115200)

# homo = Get_Position.homo
rtp = Real_Time_Predict()
rtp.create_HogDescriptor()

convert = Get_Position.World()
newcammtx = Get_Position.newcammtx

# Function _____________________________________________________________________________________________________________
def camera(pantilt):
    #CAMER.PAN(pantilt[0])
    #CAMER.TILT(pantilt[1])
    #CAMER.WaitFinish([1,9])
    pass

def inv_pantilt(pantilt):
    result = pantilt[::-1].copy()
    for i in range(len(result)):
        result[i][0] = result[i][0]*-1
    return result[:7]

def cmKhong(position):
    KHONG.SetPosition(position)
    time.sleep(0.1)
    #KHONG.WaitFinish()
    #pass

def cam_clf(DATA_PACK,brl,pt):
    rtp.create_camera_instance(0)

    #open camera and read model and predict get centroid of picture and 4 points of conner
    cardlist, midpointlist, cornerlist, realworldlist = rtp.one_time()

    #get_homo, put pantile's list by pan is q1 and tilt is q2 , blr is scene ('l', 'r', 'blbr')
    homo = blr.get_homo(q1_=pt[0], q2_=pt[1], blr_=brl)[1]

    # realworldlist is the function that convert centroid of picture to centroid of picture with respect to world coordinate
    realworldlist = convert_pos(midpointlist, newcammtx, homo)

    # same as realworldlist but it using 4 points of conner
    cornerworldlist = convert_pos(cornerlist, newcammtx, homo, mode=0, inverse=False)

    # data_pack include 1.cardlist is class og each card, 2.midpointlist is useless, 3.cornerworldlist is cornerworldlist
    #                   4.realworldlist is realworldlist, 5.data_pack (it will send to MATLAB later) 6.parameter that tell scene
    data_pack = pack_data(cardlist, midpointlist, cornerworldlist, realworldlist, DATA_PACK, brl)
    rtp.release_camera_instance()
    return data_pack

# Variable _____________________________________________________________________________________________________________
STATE = 1
CARD_POSITION = []
PANTILT = [[-50, -70],
           [90, -30], [90, -10], [90, 10], [90, 25],
           [75, 22], [70, 10], [70, 0], [70,-10], [70, -20],
           [60, -20], [60, -10], [70, 0], [70, 10], [70, 22]]

#INV_PANTILT = inv_pantilt(PANTILT)

BASE_PANTILT = [[-35, -50], [-17, -60], [0, -59], [17, -60], [35, -50]]

# Const. Position
HOME = [0,0,0,0,0,0,0]
LEFT = [270,0,0,0,0,0,0]
FRONT = [180,0,0,0,0,0,0]
RIGHT = [90,0,0,0,0,0,0]

'''
# Loop camera(pan,tilt),classify, position _____________________________________________________________________________
T_CLF = time.time()
for PT in PANTILT:
    if STATE == 1:
        # 1. Rotate J1 90 degree
        #cmKhong(RIGHT)
        # 2. Pan CAM (-50,-70)
        #camera(PT)
        # 3. Predict & Position
        #CARD_POSITION = cam_clf(CARD_POSITION,'br',PT)
        pass
    else:
        # 4. Rotate J1 90 degree
        if STATE == 2:
            cmKhong(FRONT)
        # 5. Pan CAM [95,-30],[95,-10],[95,15],[95,22],[70,22],[70,-10],[70,-30]
        camera(PT)
        # 6. Predict & Position
        CARD_POSITION = cam_clf(CARD_POSITION,'r',PT)
    STATE += 1

for PT in INV_PANTILT:
    if STATE != 16:
        # 7. Pan CAM [-95,-30],[-95,-10],[-95,15],[-95,22],[-70,22],[-70,-10],[-70,-30]
        camera(PT)
        # 8. Predict & Position
        CARD_POSITION = cam_clf(CARD_POSITION,'l',PT)
    else:
        # 9. Rotate J1 90 degree
        cmKhong([270,0,0,0,0,0,0])
        # 10. Pan CAM
        camera(PT)
        # 11. Predict & Position
        CARD_POSITION = cam_clf(CARD_POSITION,'bl',PT)
'''
# Set home position
#CAMER.PANTILT(0)
#cmKhong([0,0,0,0,0,0])
#ENDT_CLF = T_CLF - time.time()

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

# Planning (MATLAB) ____________________________________________________________________________________________________
T_PLN = time.time()
#CHIN = PathPlan(CARD_POSITION)
#PATH = CHIN.EvaluateTraject()
PATH = open('path.txt','r').read()
PATH = ast.literal_eval(PATH)

ENDT_PLN = time.time() - T_PLN
print('ENDT_PLN: ',ENDT_PLN)
# Command Khong ________________________________________________________________________________________________________
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
            cmKhong(pose)
            print('via: ',pose)
    if (path+1)%2 ==0:
        if (path+1)%4 == 0:
            KHONG.SetGrip(0)
        else:
            KHONG.SetGrip(1)

    print('Finish SUB ',path)
    time.sleep(2)

KHONG.SetGrip(0)
ENDT_LLV = time.time() - T_LLV


# In Conclusion ________________________________________________________________________________________________________
#print('ENDT_CLF: ',ENDT_CLF)
print('ENDT_LLV: ',ENDT_LLV)

