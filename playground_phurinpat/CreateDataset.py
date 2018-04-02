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
		return None

	def create_HogDescriptor(self):
		self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma, self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels)
		del self.winSize, self.blockSize, self.blockStride, self.cellSize, self.nbins, self.derivAperture, self.winSigma, self.histogramNormType, self.L2HysThreshold, self.gammaCorrection, self.nlevels

	def read_image(self,c,c_,c__):

		self.img = cv2.imread('Enc/' + str(c) + c_ + str(c__) + '.png', 0)

	def get_HOG(self,i=0,j=1):

		img_ = self.img[i*self.IMAGE_SIZE_ORI:i*self.IMAGE_SIZE_ORI+self.IMAGE_SIZE_ORI,j*self.IMAGE_SIZE_ORI:j*self.IMAGE_SIZE_ORI+self.IMAGE_SIZE_ORI].copy()
		img_ = cv2.resize(img_, (self.IMAGE_SIZE, self.IMAGE_SIZE))

		hog_ = self.hog.compute(img_).reshape(-1)
		return hog_

	def get_Histogram(self,i=0,j=1):
		row, col = np.zeros(80), np.zeros(80)
		img_ = self.img[i*self.IMAGE_SIZE_ORI:i*self.IMAGE_SIZE_ORI+self.IMAGE_SIZE_ORI,j*self.IMAGE_SIZE_ORI:j*self.IMAGE_SIZE_ORI+self.IMAGE_SIZE_ORI].copy()
		_, bw = cv2.threshold(img_, 127, 255, cv2.THRESH_BINARY)
		for i in range(0, self.IMAGE_SIZE_ORI):
			for j in range(0, self.IMAGE_SIZE_ORI):
				if bw[i][j] == 0:
					row[i] += 1
					col[j] += 1
		hist = np.hstack((row, col))
		row, col = np.zeros(80), np.zeros(80)
		return hist

def create_row(hist, hog):
	colname = []

	colname.append("class")

	for i in range(1, hist+1):
		colname.append('hist'+str(i))
	for j in range(1, hog+1):
		colname.append('hog'+str(j))

	return colname

def pack_feature(hist, hog):

	feature = np.hstack((hist, hog))
	feature = np.hstack((np.nan, feature))
	return feature

def get_class(c,c_):

	if c_ == 'N':
		return c
	elif c_ == 'E':
		return c+10
	else:
		return c+20


if __name__ == '__main__':

	first_row = create_row(160, 729)
	df = pd.DataFrame(columns=first_row)
	ins = Extraction()
	ins.create_HogDescriptor()

	NUMCELL_COL = 100
	NUMCELL_ROW = 1
#loop
	for c in range(0,10):
		for c_ in ["N", "E", "T"]:
			for c__ in range(1,6):
				ins.read_image(c,c_,c__)
				for i in range(0,NUMCELL_ROW):
					for j in range(0, NUMCELL_COL):
						hog_feature = ins.get_HOG(i,j)

						hist_feature = ins.get_Histogram(i,j)

						feature = pd.Series(pack_feature(hist_feature, hog_feature), index=first_row)
						df = df.append(feature, ignore_index=True)

			df.loc[:,'class'] = df.loc[:,'class'].fillna(get_class(c,c_))
				
			
	df.loc[:, 'hist1':'hist160'] = df.loc[:, 'hist1':'hist160'] / df.loc[:, 'hist1':'hist160'].max().max()
	df.to_csv("wow.csv", index=False)

