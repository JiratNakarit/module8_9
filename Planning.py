import matlab.engine
import numpy as np
import os

import math

class PathPlan:

    def __init__(self,CARD_POSITION):
        self.CARD_POSITION = CARD_POSITION

        #### planning

        self.eng = matlab.engine.start_matlab()
        self.eng.cd(os.getcwd(), nargout=0)

        self.a, self.b, self.c = self.eng.MotionPlanning(self.CARD_POSITION,nargout=3)

    #### Traject Gen
    class Sub_Traject:
        coef = []
        ti =0.
        t = 0.
        def __init__(self, coef=[], ti=0.,t=0.):
            self.coef = coef
            self.ti = ti
            self.t = t

    def transform_angle(self,q_configuration):
        q_homeconfig = [180,90,30,90,60,90]
        q_symbol = [-1,-1,1,-1,1,-1]
        q_transform = []
        for i in range(0,6):
            q_transform.append(q_homeconfig[i] + q_symbol[i]*q_configuration[i])
        return q_transform

    def GenTraject(self):
        traject = []
        for number_traject in range(len(self.a)):
            sub = []
            for number_sub in range(len(self.a[number_traject][0])):
                co = []
                for number_joint in range(6):
                    co_ = []
                    for i in range(len(self.a[0])):
                        co_.append(self.a[number_traject][i][number_sub][number_joint])
                    co.append(co_)
                ti = self.b[number_traject][number_sub][0]
                t = self.c[number_traject][number_sub][0]
                sub.append(self.Sub_Traject(co,ti,t))
            traject.append(sub)
        return traject

    def EvaluateTraject(self):
        Traject = []
        traject = self.GenTraject()
        dt = 0.1
        time = 0
        #T_sub = []
        for p_ind in range(len(traject)):
            K = len(traject[p_ind])
            #T_time = []
            pathTraject = []
            for k in range(K):
                subTraject = []
                q = np.zeros((6,))
                c = traject[p_ind][k].coef
                t_i = traject[p_ind][k].ti
                T = traject[p_ind][k].t
                sampling_time = np.arange(t_i,T+t_i,dt)
                sampling_time = np.append(sampling_time,T+t_i)
                for t in sampling_time:
                    tau = t-t_i
                    #time += tau
                    for d in range(6):
                        q[d] = int(np.degrees(c[d][0] + c[d][1]*tau + c[d][2]*tau**2 + c[d][3]*tau**3))
                    #T_time.append(time)
                    subTraject.append(self.transform_angle(q.astype(np.int)))
                pathTraject.append(subTraject)
           #T_sub.append(T_time)
            Traject.append(pathTraject)
        return Traject

    def DurationTime(self):
        T_all = self.c
        run_time = []
        time = 0
        for i in range(len(T_all)):
            path_i = T_all[i]
            time_i = []
            for k in range(len(path_i)):
                time += path_i[k][0]
                time_i.append(time)
            run_time.append(time_i)
        return run_time

if __name__ == "__main__":
    pi = math.pi
    card_list_ = [[[650.,0.,20.],[pi,0.,0.],0.],
                [[600.,-480.,800.],[pi/2,-pi/2,0.],13.]]

    chin = PathPlan(card_list_)

    path = chin.EvaluateTraject()
    f = open('path.txt','w')
    f.write(repr(path))
