#python2
import sys
# print(sys.path)
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
	sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
	print("ROS! removed")
else:
	print("Fine")
import cv2
import pandas as pd
import numpy as np
import copy
import os


class Extraction():

	''' Make Dataset '''

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

	def crop_(self, plate_img, i, j):
		top_bot = []
		left_right = []
		kernel = np.ones((3, 3), np.uint8)
		img = plate_img.copy()
		img_ = copy.copy(img[i * self.IMAGE_SIZE_ORI:i * self.IMAGE_SIZE_ORI + self.IMAGE_SIZE_ORI,
		                 j * self.IMAGE_SIZE_ORI:j * self.IMAGE_SIZE_ORI + self.IMAGE_SIZE_ORI])
		img_ = cv2.adaptiveThreshold(img_,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,1)
		invt = cv2.bitwise_not(copy.copy(img_))
		invt = cv2.dilate(invt, kernel, iterations=1)

		ret, contours, _ = cv2.findContours(invt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		for i in contours:
			if cv2.contourArea(i) <= 10:
				pass
			else:
				x, y, w, h = cv2.boundingRect(i)
				top_bot.append(y)
				top_bot.append(y + h)
				left_right.append(x)
				left_right.append(x + w)

		word_crop = self.img[min(top_bot):max(top_bot), min(left_right):max(left_right)].copy()
		word_crop = cv2.resize(word_crop, (50, 50))

		# cv2.imshow("img", img_)
		# cv2.imshow("invt", invt)
		# cv2.imshow("Crop", word_crop)
		# cv2.waitKey(1)
		# cv2.destroyAllWindows()
		return word_crop

	def create_HogDescriptor(self):
		self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma, self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels)
		del self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma, self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels

	def read_image(self,c,c_):

		self.img = cv2.imread('Enc_/' + str(c) +"_"+ str(c_) + '.png', 0)

	def get_HOG(self, img_):

		hog_ = self.hog.compute(img_).reshape(-1)
		return hog_

	def get_Histogram(self, img_):
		row, col = np.zeros(self.IMAGE_SIZE), np.zeros(self.IMAGE_SIZE)

		_, bw = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)
		for i in range(0, self.IMAGE_SIZE):
			for j in range(0, self.IMAGE_SIZE):
				if bw[i][j] == 0:
					row[i] += 1
					col[j] += 1
		hist = np.hstack((row, col))
		row, col = np.zeros(self.IMAGE_SIZE), np.zeros(self.IMAGE_SIZE)
		return hist


def create_row(hist, hog):
	colname = []

	colname.append("class")

	for i in range(1, hist+1):
		colname.append('hist'+str(i))
	for j in range(1, hog+1):
		colname.append('hog'+str(j))

	return colname

def pack_feature(hog, hist=None):

	feature = np.hstack((hist, hog))
	feature = np.hstack((np.nan, feature))
	return feature


if __name__ == '__main__':

	first_row = create_row(100, 729)
	df = pd.DataFrame(columns=first_row)
	ins = Extraction()
	ins.create_HogDescriptor()

	# ins.read_image(9, 1)
	# crop_im = ins.crop_(ins.img, 0, 0)
	# hog_feature = ins.get_HOG(img_=crop_im)
	# print(hog_feature.shape)
	# hist_feature = ins.get_Histogram(img_=crop_im)
	# print(hist_feature.shape)
	# print(len(first_row))


	NUMCELL_COL = 50
	NUMCELL_ROW = 10

	for c in range(0,30):
		for c_ in range(1,7):
			ins.read_image(c,c_)
			for i in range(0,NUMCELL_ROW):
				for j in range(0, NUMCELL_COL):

					crop_im = ins.crop_(ins.img, i, j)
					hog_feature = ins.get_HOG(img_=crop_im)
					hist_feature = ins.get_Histogram(img_=crop_im)
					
					feature = pd.Series(pack_feature(hog=hog_feature,hist=hist_feature), index=first_row)
					df = df.append(feature, ignore_index=True)
			print(str(c) + str(c_))
		df.loc[:,'class'] = df.loc[:,'class'].fillna(c)

	df.loc[:, 'hist1':'hist100'] = df.loc[:, 'hist1':'hist100'] / 50
	df.to_csv("news_crop_Dataset_nohist.csv", index=False)
	print("Done")
        