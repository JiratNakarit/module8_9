import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm

class World:
	def calculate_World_coor(self,u,v,newcammtx,homo):

		mat_a = np.eye(4)

		r_t_mtx = inv(np.matrix(homo))

		cam_point = r_t_mtx @ np.matrix([0,0,0,1]).T

		m_ = np.matrix([u,v,1,1]).T

		mat_a[0:3,0:3] = newcammtx
		mat_a = inv(np.matrix(mat_a))
		
		x_y_z = (r_t_mtx @ mat_a) @ m_

		unit_vector = (x_y_z[0:3]-cam_point[0:3]) / norm(x_y_z[0:3]-cam_point[0:3])

		lambda_ = -cam_point[2] / unit_vector[2]

		u_v = (mat_a @ r_t_mtx) @ x_y_z

		final_u_v = cam_point[0:3] + (float(lambda_) * unit_vector)

		return x_y_z, final_u_v

	def calculate_World_coor_reverse(self,x,y,newcammtx,homo):

		mat_a = np.eye(4)

		r_t_mtx = np.matrix(homo)

		cam_point = []

		m_ = np.matrix([x,y,1,1]).T

		mat_a[0:3,0:3] = newcammtx
		mat_a = np.matrix(mat_a)
		
		x_y_z = np.matrix([x,y,0,1]).T

		u_v = (mat_a @ r_t_mtx) @ x_y_z


		return x_y_z, u_v/u_v[2]

# dat = np.load('calibresult.npz')
dat1 = np.load('calibresult1.npz')
# dat = np.load('output_matrix.npz')
# print(dat.camera_matrix)

newcammtx = dat1['newcameramtx']
# print(dat['newcameramtx'])
# print(dat1['newcameramtx'])
# newcammtx = dat['camera_matrix']
# print(dat['mtx'])

# print(newcammtx)

# newcammtx[0][2] = 302
# newcammtx[1][2] = 260

homo = np.eye(4)
	
# homo[1:3,:] = [[0, -0.7071, -0.7071, 0], [0, 0.7071, -0.7071, 0.85]]
homo[0:3,3] = [0, 0, 1.3]

if __name__ == "__main__":

	world = World()

	homo = np.eye(4)
	
	homo[1:3,:] = [[0, -0.7071, -0.7071, 0], [0, 0.7071, -0.7071, 0.85]]

	u_v_re = world.calculate_World_coor_reverse(100, 0, newcammtx, homo)

	print('calculate_World_coor_reverse : \n\tXYZ : \n{0}\n\tUV : \n{1}\n_______________________'.format(u_v_re[0], u_v_re[1]))

	x_y = world.calculate_World_coor(int(u_v_re[1][0]), int(u_v_re[1][1]), newcammtx, homo)

	print('calculate_World_coor : \n\tXYZ : \n{0}\n\tUV : \n{1}\n_______________________'.format(x_y[0], x_y[1]))
