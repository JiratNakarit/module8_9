# B,L,R

import numpy as np

def get_homo(q1_=0,q2_=0,blr_='l'):

	pi = np.pi
	if blr_ == 'bl':
		q1 = -pi/2
	elif blr_ == 'br':
		q1 = pi/2
	elif blr_ == 'l' or blr_ == 'r':
		q1 = pi
		
	qc1 = np.radians(q1_)
	qc2 = np.radians(q2_)
	lc1 = 202.5 / 1000
	hc1 = 249 / 1000
	hc2 = 19 / 1000
	lc2 = 14.75 / 1000
	hc3 = 82 / 1000
	lc3 = 33.5 / 1000
	l1 = 250.5 / 1000

	B = [[-np.sin(q1 - qc1), -np.cos(q1 - qc1) * np.sin(qc2), np.cos(q1 - qc1) * np.cos(qc2), - lc1 * np.cos(q1) - lc2 * np.cos(q1) * np.cos(qc1) - lc2 * np.sin(q1) * np.sin(qc1) - lc3 * np.cos(q1) * np.cos(qc1) * np.cos(qc2) - hc3 * np.cos(q1) * np.cos(qc1) * np.sin(qc2) - lc3 * np.cos(qc2) * np.sin(q1) * np.sin(qc1) - hc3 * np.sin(q1) * np.sin(qc1) * np.sin(qc2) - 300 / 1000],
		[np.cos(q1 - qc1), -np.sin(q1 - qc1) * np.sin(qc2), np.sin(q1 - qc1) * np.cos(qc2), lc2 * np.cos(q1) * np.sin(qc1) - lc1 * np.sin(q1) - lc2 * np.cos(qc1) * np.sin(q1) + lc3 * np.cos(q1) * np.cos(qc2) * np.sin(qc1) - lc3 * np.cos(qc1) * np.cos(qc2) * np.sin(q1) + hc3 * np.cos(q1) * np.sin(qc1) * np.sin(qc2) - hc3 * np.cos(qc1) * np.sin(q1) * np.sin(qc2)],
		[0, np.cos(qc2), np.sin(qc2), hc1 + hc2 + l1 + hc3 * np.cos(qc2) - lc3 * np.sin(qc2) - 26 / 1000],
		[0, 0, 0, 1]]


	L = [[-np.sin(q1 - qc1), -np.cos(q1 - qc1) * np.sin(qc2),  np.cos(q1 - qc1) * np.cos(qc2), - lc1 * np.cos(q1) - lc2 * np.cos(q1) * np.cos(qc1) - lc2 * np.sin(q1) * np.sin(qc1) - lc3 * np.cos(q1) * np.cos(qc1) * np.cos(qc2) - hc3 * np.cos(q1) * np.cos(qc1) * np.sin(qc2) - lc3 * np.cos(qc2) * np.sin(q1) * np.sin(qc1) - hc3 * np.sin(q1) * np.sin(qc1) * np.sin(qc2) - 300 / 1000],
		[0, np.cos(qc2), np.sin(qc2), hc1 + hc2 + l1 + hc3 * np.cos(qc2) - lc3 * np.sin(qc2) - 500 / 1000],
		[ -np.cos(q1 - qc1),  np.sin(q1 - qc1) * np.sin(qc2), -np.sin(q1 - qc1) * np.cos(qc2), lc1 * np.sin(q1) - lc2 * np.cos(q1) * np.sin(qc1) + lc2 * np.cos(qc1) * np.sin(q1) - lc3 * np.cos(q1) * np.cos(qc2) * np.sin(qc1) + lc3 * np.cos(qc1) * np.cos(qc2) * np.sin(q1) - hc3 * np.cos(q1) * np.sin(qc1) * np.sin(qc2) + hc3 * np.cos(qc1) * np.sin(q1) * np.sin(qc2) + 476 / 1000],
		[0, 0, 0, 1]]


	R = [[np.sin(q1 - qc1),  np.cos(q1 - qc1) * np.sin(qc2), -np.cos(q1 - qc1) * np.cos(qc2), lc1 * np.cos(q1) + lc2 * np.cos(q1) * np.cos(qc1) + lc2 * np.sin(q1) * np.sin(qc1) + lc3 * np.cos(q1) * np.cos(qc1) * np.cos(qc2) + hc3 * np.cos(q1) * np.cos(qc1) * np.sin(qc2) + lc3 * np.cos(qc2) * np.sin(q1) * np.sin(qc1) + hc3 * np.sin(q1) * np.sin(qc1) * np.sin(qc2) + 300 / 1000],
		[0, np.cos(qc2), np.sin(qc2), hc1 + hc2 + l1 + hc3 * np.cos(qc2) - lc3 * np.sin(qc2) - 500 / 1000],
		[np.cos(q1 - qc1), -np.sin(q1 - qc1) * np.sin(qc2),  np.sin(q1 - qc1) * np.cos(qc2), lc2 * np.cos(q1) * np.sin(qc1) - lc1 * np.sin(q1) - lc2 * np.cos(qc1) * np.sin(q1) + lc3 * np.cos(q1) * np.cos(qc2) * np.sin(qc1) - lc3 * np.cos(qc1) * np.cos(qc2) * np.sin(q1) + hc3 * np.cos(q1) * np.sin(qc1) * np.sin(qc2) - hc3 * np.cos(qc1) * np.sin(q1) * np.sin(qc2) + 476 / 1000],
		[0, 0, 0, 1]]

	if blr_ == 'l':
		return L
	elif blr_ == 'r':
		return R
	elif blr_ == 'bl' or blr_ == 'br':
		return B



# input q1 = pi, qc1 <- >+, qc2 down+ up-

if __name__ == '__main__':
	blr = get_homo(q1_=90, q2_=0, blr_='l')
	for i in blr[1]:
		print(i)
