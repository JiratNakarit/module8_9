import cv2
import numpy as np

path = '/media/phurinpat/645E699C5E69682E/Users/PHURINPAT/Desktop/FirstSample90000Pic/'
dst_path = 'Enc_/'

n_ = range(0, 30)
f_ = range(1, 7)

def main(plate_num, fontnum):
	img = cv2.imread(path + str(plate_num) + "_" + str(fontnum) + ".png", 0)
	img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY, 21,5)
	for i_ in range(0, 10):
		for j_ in range(0, 50):
			im = img[100*i_:(100*i_)+100:, 100*j_:(100*j_)+100]
			im = im[10:90,10:90]
			if j_:
				dst_ = np.hstack((dst_,im))
			else:
				dst_ = im
		if i_:
			dst__ = np.vstack((dst_,dst__))
		else:
			dst__ = dst_
	cv2.imwrite(dst_path + str(plate_num) + "_" + str(fontnum) + ".png", dst__)

	print("Write"+dst_path + str(plate_num) + "_" + str(fontnum) + ".png")

for i in n_:
		for k in f_:
			main(i, k)