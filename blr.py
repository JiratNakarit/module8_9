import numpy as np


def get_homo(q1_=0, q2_=0, blr_=''):
	q2_ *= -1
	q1_ *= -1
	pi = np.pi
	if blr_ == 'bl':
		q1 = -pi / 2
	elif blr_ == 'br':
		q1 = pi / 2
	elif blr_ == 'l' or blr_ == 'r' or blr_ == 'b':
		q1 = pi

	cos = np.cos
	sin = np.sin

	qc1 = np.radians(q1_)
	qc2 = np.radians(q2_)
	lc1 = 202.5 / 1000
	hc1 = 249 / 1000
	hc2 = 19 / 1000
	lc2 = 14.75 / 1000
	hc3 = 82 / 1000
	lc3 = 33.5 / 1000
	l1 = 250.5 / 1000
	dx = 300 / 1000
	dy = 474 / 1000
	dz = 500 / 1000

	B = [[-sin(q1 - qc1), cos(q1 - qc1), 0, lc1 * sin(qc1) - dx * sin(q1 - qc1)],
		 [-cos(q1 - qc1) * sin(qc2), -sin(q1 - qc1) * sin(qc2), cos(qc2),
		  - hc3 - hc1 * cos(qc2) - hc2 * cos(qc2) - l1 * cos(qc2) - lc2 * sin(qc2) - lc1 * cos(qc1) * sin(
			  qc2) - dx * cos(q1) * cos(qc1) * sin(qc2) - dx * sin(q1) * sin(qc1) * sin(qc2)],
		 [cos(q1 - qc1) * cos(qc2), sin(q1 - qc1) * cos(qc2), sin(qc2),
		  lc3 + lc2 * cos(qc2) - hc1 * sin(qc2) - hc2 * sin(qc2) - l1 * sin(qc2) + lc1 * cos(qc1) * cos(qc2) + dx * cos(
			  q1) * cos(qc1) * cos(qc2) + dx * cos(qc2) * sin(q1) * sin(qc1)],
		 [0, 0, 0, 1]]

	L = [[-sin(q1 - qc1), 0, -cos(q1 - qc1), lc1 * sin(qc1) + dy * cos(q1 - qc1) - dx * sin(q1 - qc1)],
		 [-cos(q1 - qc1) * sin(qc2), cos(qc2), sin(q1 - qc1) * sin(qc2),
		  dz * cos(qc2) - hc3 - hc1 * cos(qc2) - hc2 * cos(qc2) - l1 * cos(qc2) - lc2 * sin(qc2) - lc1 * cos(qc1) * sin(
			  qc2) - dx * cos(q1) * cos(qc1) * sin(qc2) + dy * cos(q1) * sin(qc1) * sin(qc2) - dy * cos(qc1) * sin(
			  q1) * sin(qc2) - dx * sin(q1) * sin(qc1) * sin(qc2)],
		 [cos(q1 - qc1) * cos(qc2), sin(qc2), -sin(q1 - qc1) * cos(qc2),
		  lc3 + lc2 * cos(qc2) + dz * sin(qc2) - hc1 * sin(qc2) - hc2 * sin(qc2) - l1 * sin(qc2) + lc1 * cos(qc1) * cos(
			  qc2) + dx * cos(q1) * cos(qc1) * cos(qc2) - dy * cos(q1) * cos(qc2) * sin(qc1) + dy * cos(qc1) * cos(
			  qc2) * sin(q1) + dx * cos(qc2) * sin(q1) * sin(qc1)],
		 [0, 0, 0, 1]]

	R = [[sin(q1 - qc1), 0, cos(q1 - qc1), lc1 * sin(qc1) - dy * cos(q1 - qc1) - dx * sin(q1 - qc1)],
		 [cos(q1 - qc1) * sin(qc2), cos(qc2), -sin(q1 - qc1) * sin(qc2),
		  dz * cos(qc2) - hc3 - hc1 * cos(qc2) - hc2 * cos(qc2) - l1 * cos(qc2) - lc2 * sin(qc2) - lc1 * cos(qc1) * sin(
			  qc2) - dx * cos(q1) * cos(qc1) * sin(qc2) - dy * cos(q1) * sin(qc1) * sin(qc2) + dy * cos(qc1) * sin(
			  q1) * sin(qc2) - dx * sin(q1) * sin(qc1) * sin(qc2)],
		 [-cos(q1 - qc1) * cos(qc2), sin(qc2), sin(q1 - qc1) * cos(qc2),
		  lc3 + lc2 * cos(qc2) + dz * sin(qc2) - hc1 * sin(qc2) - hc2 * sin(qc2) - l1 * sin(qc2) + lc1 * cos(qc1) * cos(
			  qc2) + dx * cos(q1) * cos(qc1) * cos(qc2) + dy * cos(q1) * cos(qc2) * sin(qc1) - dy * cos(qc1) * cos(
			  qc2) * sin(q1) + dx * cos(qc2) * sin(q1) * sin(qc1)],
		 [0, 0, 0, 1]]

	if blr_ == 'l':
		return L
	elif blr_ == 'r':
		return R
	elif blr_ == 'bl' or blr_ == 'br' or blr_ == 'b':
		return B


if __name__ == '__main__':
	blr = get_homo(q1_=90, q2_=0, blr_='r')
	blr = np.linalg.inv(blr)
	for i in blr:
		print(i)