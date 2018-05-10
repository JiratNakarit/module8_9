import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    print("Path removed")
else:
    pass
import cv2
import numpy as np

camera = cv2.VideoCapture(0)
c = 1

while 1:
    ret, frame = camera.read()
    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        cv2.destroyAllWindows()
        camera.release()
        break
    elif k == ord("c"):
        cv2.imwrite("cm" + str(c) + ".png", frame)
        print("save: "+ "cm" + str(c) + ".png")
        c+=1
