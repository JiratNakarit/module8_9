# Import module ________________________________________________________________________________________________________
import time

# Import class _________________________________________________________________________________________________________
from Dynamixel import *
from Bord import *
from RealtimePredict import *
from blr import *
from Get_Position import *
from Planning import PathPlan

# Call class ___________________________________________________________________________________________________________
CAMER = Dynamixel('COM',1000000)
#KHONG = Board('COM',115200)

# homo = Get_Position.homo
rtp = Real_Time_Predict()
rtp.create_camera_instance(0)
rtp.create_HogDescriptor()

convert = Get_Position.World()
newcammtx = Get_Position.newcammtx

# Function _____________________________________________________________________________________________________________
def camera(pantilt):
    CAMER.PAN(pantilt[0])
    CAMER.TILT(pantilt[1])

def inv_pantilt(pantilt):
    result = pantilt[::-1].copy()
    for i in range(len(result)):
        result[i][0] = result[i][0]*-1
    return result[:7]

def cmKhong(position):
    #KHONG.SetPosition(position)
    #KHONG.WaitFinish()
    pass
    #return KHONG.GetPosition()

def cam_clf(DATA_PACK,brl,pt):
    rtp.release_camera_instance()
    rtp.create_camera_instance(0)

    #open camera and read model and predict get centroid of picture and 4 points of conner
    cardlist, midpointlist, cornerlist, realworldlist = rtp.one_time()

    #get_homo, put pantile's list by pan is q1 and tilt is q2 , blr is scene ('l', 'r', 'blbr')
    homo = blr.get_homo(q1_=pt[0], q2_=pt[1], blr_=brl)

    # realworldlist is the function that convert centroid of picture to centroid of picture with respect to world coordinate
    realworldlist = convert_pos(midpointlist, newcammtx, homo)

    # same as realworldlist but it using 4 points of conner
    cornerworldlist = convert_pos(cornerlist, newcammtx, homo, mode=0, inverse=False)

    # data_pack include 1.cardlist is class og each card, 2.cornerworldlist is cornerworldlist
    #                   3.realworldlist is realworldlist, 4.data_pack (it will send to MATLAB later) 5.parameter that tell scene
    data_pack = pack_data(cardlist, cornerworldlist, realworldlist, DATA_PACK, brl)
    return data_pack

# Variable _____________________________________________________________________________________________________________
STATE = 1
CARD_POSITION = []
PANTILT = [[-50,-70],
           [90,-30],[90,-10],[90,10],[90,25],
           [75,22],[70,10],[70,0],[70,-10],[70,-20],
           [60,-20],[60,-10],[70,0],[70,10],[70,22]]
INV_PANTILT = inv_pantilt(PANTILT)
data_pack = []

# Const. Position
HOME = [0,0,0,0,0,0,0]
LEFT = [270,0,0,0,0,0,0]
FRONT = [180,0,0,0,0,0,0]
RIGHT = [90,0,0,0,0,0,0]

# Loop camera(pan,tilt),classify, position _____________________________________________________________________________
T_CLF = time.time()
for PT in PANTILT:
    if STATE == 1:
        # 1. Rotate J1 90 degree
        cmKhong(RIGHT)
        # 2. Pan CAM (-50,-70)
        camera(PT)
        # 3. Predict & Position
        data_pack = cam_clf(data_pack,'br',PT)
        CARD_POSITION.append(data_pack)
    else:
        # 4. Rotate J1 90 degree
        if STATE == 2:
            cmKhong(FRONT)
        # 5. Pan CAM [95,-30],[95,-10],[95,15],[95,22],[70,22],[70,-10],[70,-30]
        camera(PT)
        # 6. Predict & Position
        data_pack = cam_clf(data_pack,'r',PT)
        CARD_POSITION.append(data_pack)
    STATE += 1

for PT in INV_PANTILT:
    if STATE != 16:
        # 7. Pan CAM [-95,-30],[-95,-10],[-95,15],[-95,22],[-70,22],[-70,-10],[-70,-30]
        camera(PT)
        # 8. Predict & Position
        data_pack = cam_clf(data_pack,'l',PT)
        CARD_POSITION.append(data_pack)
    else:
        # 9. Rotate J1 90 degree
        cmKhong([270,0,0,0,0,0,0])
        # 10. Pan CAM
        camera(PT)
        # 11. Predict & Position
        data_pack = cam_clf(data_pack,'bl',PT)
        CARD_POSITION.append(data_pack)
'''
# Set home position
CAMER.PANTILT(0)
cmKhong([0,0,0,0,0,0])
ENDT_CLF = T_CLF - time.time()

# Planning (MATLAB) ____________________________________________________________________________________________________
T_PLN = time.time()
CHIN = PathPlan(CARD_POSITION)
PATH = CHIN.EvaluateTraject()
ENDT_PLN = T_PLN - time.time()

# Command Khong ________________________________________________________________________________________________________
T_LLV = time.time()
#Traject
for Tpath in range(len(PATH[:])):
    #Sub Traject
    for STpath in PATH[Tpath]:
        cmKhong(STpath)
    #gripper
    if (Tpath+1)%4 == 0:
        KHONG.SetGrip(0)
ENDT_LLV = T_LLV - time.time()


# In Conclusion ________________________________________________________________________________________________________
print('ENDT_CLF: ',ENDT_CLF)
print('ENDT_PLN: ',ENDT_PLN)
print('ENDT_LLV: ',ENDT_LLV)
'''
