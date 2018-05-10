import serial
import time


class Dynamixel:

    def __init__(self,comPort,baudRate=1000000,timeOut=0.1):
        self.comPort = comPort
        self.baudRate = baudRate
        self.timeOut = timeOut
        self.serial = serial.Serial(self.comPort)
        self.serial.baudrate = self.baudRate

    def Angle2Position(self,ANGLE):
        FACTOR = 3.41
        return int(((ANGLE)+150)*FACTOR)

    def Position2Angle(self,POSITION):
        FACTOR = 3.41
        return int((POSITION/FACTOR)-150)*-1

    def WriteData(self,INSTRUCTION_PACKET):
        INSTRUCTION_PACKET.insert(3, len(INSTRUCTION_PACKET)-2)
        CHKSUM = 0
        for i in range(2, len(INSTRUCTION_PACKET)):
            CHKSUM += INSTRUCTION_PACKET[i]
        CHKSUM %= 256
        INSTRUCTION_PACKET.append(255-CHKSUM)
        self.serial.write(INSTRUCTION_PACKET)

    def ReadData(self):
        STATUS_PACKET = []
        start = time.clock()
        while time.clock() - start <= self.timeOut:
            while self.serial.inWaiting():
                STATUS_PACKET.append(ord(self.serial.read()))
        return STATUS_PACKET

    def set_position(self, ID, GOAL=0, SPEED=140):

        GOAL = self.Angle2Position(GOAL)

        H_GOAL, L_GOAL = divmod(GOAL, 256)
        H_SPEED, L_SPEED = divmod(SPEED, 256)
        if type(ID) is int:
            PACKET = [255, 255, ID, 3, 30, L_GOAL, H_GOAL, L_SPEED, H_SPEED]
            self.WriteData(PACKET)
        else:
            for id in ID:
                PACKET = [255, 255, id, 3, 30, L_GOAL, H_GOAL, L_SPEED, H_SPEED]
                self.WriteData(PACKET)
        STATUS = self.ReadData()

    def GetStatus(self,ID):
        if type(ID) is int:
            PACKET = [255,255,ID,2,46,1]
            self.WriteData(PACKET)
            packet = self.ReadData()
            return packet[5]
        else:
            result = []
            for id in ID:
                PACKET = [255,255,id,2,46,1]
                self.WriteData(PACKET)
                res = self.ReadData()
                result.append(res[5])
            return result

    def WaitFinish(self,ID):
        if type(ID) is int:
            while True:
                if self.GetStatus(ID) == 0:
                    break
        else:
            while True:
                if 1 not in self.GetStatus(ID):
                    break

    def get_position(self, ID):
        if type(ID) is int:
            PACKET = [255, 255, ID, 2, 36, 2]
            self.WriteData(PACKET)
            res = self.ReadData()
            print('res',res)
            result = res[6]*256+res[5]
            return self.Position2Angle(result)
        else:
            result = []
            for id in ID:
                PACKET = [255, 255, id, 2, 36, 2]
                self.WriteData(PACKET)
                res = self.ReadData()
                result.append(self.Position2Angle(res[6] * 256 + res[5]))
            return result

    def PAN(self, GOAL):
        self.set_position(9, GOAL=GOAL*-1)

    def TILT(self, GOAL):
        self.set_position(1, GOAL=GOAL*-1)

    def PANTILT(self, GOAL):
        self.set_position([1,9], GOAL=GOAL*-1)

if __name__ == "__main__":
    AX = Dynamixel('COM3',1000000)

    AX.PANTILT(0)
    AX.WaitFinish([1,9])
    AX.PANTILT(60)
    AX.WaitFinish([1,9])
    AX.PANTILT(0)
    AX.WaitFinish([1,9])
    AX.PANTILT(-60)
    AX.WaitFinish([1,9])
    AX.PANTILT(0)
    AX.WaitFinish([1,9])
