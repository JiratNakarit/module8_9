import serial
import time

class Board:

    def __init__(self,comport,buadrate,timeout=0.1):
        self.comport = comport
        self.buadrate = buadrate
        self.timeout = timeout
        self.serial = serial.Serial(self.comport)
        self.serial.baudrate = self.buadrate
        self.status = 0

    def WriteData(self,INSTRUCTION_PACKET):
        self.serial.flush()
        self.serial.write(INSTRUCTION_PACKET)

    def ReadData(self):
        STATUS_PACKET = []
        start = time.clock()
        while time.clock() - start <= self.timeout:
            while self.serial.inWaiting():
                STATUS_PACKET.append(ord(self.serial.read()))
        return STATUS_PACKET

    def GetStatus(self,ID=0):
        PACKET = self.GenPacket(0,[2,7,1])
        self.WriteData(PACKET)
        packet = self.ReadData()
        try: return packet[5]
        except: return 1

    def WaitFinish(self):
        pass
        while True:
            status = self.GetStatus()
            print(status)
            if status == 0:
                break

    def GenPacket(self,ID,PACKET):
        for i in range(2):
            PACKET.insert(0,255)
        PACKET.insert(2,ID)
        PACKET.insert(3, len(PACKET)-2)
        CHKSUM = 0
        for i in range(2, len(PACKET)):
            CHKSUM += PACKET[i]
        CHKSUM %= 256
        PACKET.append(255-CHKSUM)
        return PACKET

    def SetPosition(self,GOAL,ID=0):

        if len(GOAL) > 6:
            raise NameError('There are too many packet')
        elif len(GOAL) < 6:
            raise NameError('There are not enough packet')

        GOAL.insert(0,1)
        GOAL.insert(0,3)
        packet = self.GenPacket(ID,GOAL)
        self.WriteData(packet)
        time.sleep(0.18)

    def GetPosition(self,ID=0):
        data = []
        packet = self.GenPacket(ID,[2,1,6])
        self.WriteData(packet)
        res = self.ReadData()
        for i in range(len(res)):
            data.append(res[i])
        return data

    def SetGrip(self,STATUS):
        if STATUS in [0,1]:
            packet = self.GenPacket(0,[3,0,STATUS])
            print(packet)
            self.WriteData(packet)
        else:
            raise NameError('Gripper Has Only Status 0 (Closed) or 1 (Open)')
    def GetGrip(self,ID=0):
        packet = self.GenPacket(ID,[2,7,1])
        self.WriteData(packet)
        res = self.ReadData()
        return res

if __name__ == '__main__':

    KHONG = Board('COM4',115200)


    # KHONG.SetPosition([0,0,0,90,90,90])

    #while True:
    #for i in [40,65,75,85,95,0]:
    #pos = [90,90,90,90,90,90]
        #print(pos)
    #KHONG.SetPosition(pos)
        #print(KHONG.GetStatus())
        #a = KHONG.GetStatus()
        #print(KHONG.GetPosition())
    KHONG.SetGrip(1)
    #KHONG.WaitFinish()
        #print(a)
        #if a == 0:
            #break

    #KHONG.SetPosition([0,0,0,45,45,0])

    #print(KHONG.GetPosition())
