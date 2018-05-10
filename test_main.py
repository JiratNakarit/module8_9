from Main import *
import math
import ast

while True:
    key = input('Select Mode: ')
    if key == 'exit()':
        break
    elif key == 'cmd()':
        while True:
            print('1. Joint')
            print('2. Dynamixel')
            mode = input('Select NoMode: ')
            if mode == 'exit()':
                break
            elif mode == '1':
                KHONG = Main(nomodeset=1)
                while True:
                    val = input('Type your position')
                    if val == 'exit()':
                        break
                    val = ast.literal_eval(val)
                    if type(val) is list:
                        KHONG.cmKhong(val)
                    else:
                        print('ERROR!!!!!!!!!!!!!!!!')
            elif mode == '2':
                KHONG = Main(nomodeset=2)
                while True:
                    val = input('Type your position')
                    if val == 'exit()':
                        break
                    val = ast.literal_eval(val)
                    if type(val) is list:
                        KHONG.camera(val)
                    else:
                        print('ERROR!!!!!!!!!!!!!!!!')
            else:
                print('ERROR!!!!!!!!!!!!!!!!')
    elif key == 'test()':
        while True:
            print('1. Run a little bit')
            print('2. Run full sequence')
            mode = input('Select NoMode: ')
            KHONG = Main(nomodeset=0)
            if mode == 'exit()':
                break
            elif mode == '1':
                PATH = open('path.txt','r').read()
                PATH = ast.literal_eval(PATH)
            elif mode == '2':
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
                PATH, T_path = KHONG.Step2PathPlan(CARD_POSITION)
            else:
                print('ERROR!!!!!!!!!!!!!!!!')

            T_cmd = Sequen.Step3CommandKhong(Path)

            print('Summary')
            print('1. Time for find card: ',T_1)
            print('2. Time for Planning : ',T_2)
            print('3. Time for Running  : ',T_3)

    else:
        print('ERROR!!!!!!!!!!!!!!!!')
