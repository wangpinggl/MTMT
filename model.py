'''
Multi-Task Learning Codes
'''
import copy
import numpy as np
# from GPlot import loss_plot


def LF1Projection(B, parB):
    tA = copy.deepcopy(B)
    n_task = len(B)
    n_feature = B[0].shape[0]
    for f in range(n_feature):
        nm = 0.0
        for k in range(n_task):
            nm += np.linalg.norm(B[k][f, :])**2
        nm = np.sqrt(nm)
        if nm == 0:
            for k in range(n_task):
                tA[k][f] *= 0.0
        else:
            for k in range(n_task):
                tA[k][f] = np.maximum(nm - parB, 0) / nm * B[k][f]
    return tA


def update_gradB(XtX, X, Y, B, u, parL2, parTime):
    gradB = -X.T.dot(Y+u) + XtX.dot(B)
    gradB += parL2 * B
    n_time = B.shape[1]
    for j in range(n_time):
        if j == 0:
            gradB[:, j] += parTime*(B[:, j] - B[:, j+1])
            continue
        if j == n_time-1:
            gradB[:, j] += parTime*(B[:, j] - B[:, j-1])
            continue
        gradB[:, j] += parTime*(2.0*B[:, j] - B[:, j-1] - B[:, j+1])

    return gradB


def update_funcB(X, B, Y, u, parL2, parTime):
    func = 0.0
    func += 0.5*np.linalg.norm(Y+u-X.dot(B))**2
    func += 0.5*parL2*np.linalg.norm(B)**2
    n_time = B.shape[1]
    for j in range(n_time-1):
        func += 0.5*parTime*np.linalg.norm(B[:, j]-B[:, j+1])**2

    return func


def TopUp(M):
    tM = copy.deepcopy(M)
    for k in range(M.shape[0]):
        for j in range(M.shape[1]-1):
            if tM[k, j+1] > tM[k, j]:
                tM[k, j+1] = tM[k, j]
    return tM


class mtLF1(object):
    '''
    Multi_Task Learning With LF1 norm
    '''

    def __init__(self, X, Y, S, rhoADMM=1.0, rhoL2=1.0, rhoLF1=1.0, rhoTime=1.0, max_iter=100, BB=[], MM=[], uu=[]):
        '''
        Constructor
        '''
        self.X = X
        self.Y = Y
        self.S = S
        self.init_rand = True
        if len(BB) > 0:
            self.init_rand = False
            self.B = copy.deepcopy(BB)
            self.M = copy.deepcopy(MM)
            self.u = copy.deepcopy(uu)
        self.n_task = len(X)
        self.n_time = Y[0].shape[1]
        self.n_instance = []
        for k in range(self.n_task):
            self.n_instance.append(Y[k].shape[0])
        self.n_feature = X[0].shape[1]

        self.rhoADMM = rhoADMM
        self.rhoL2 = rhoL2
        self.rhoLF1 = rhoLF1
        self.rhoTime = rhoTime
        self.max_iter = max_iter

        self.obj = []

    def init_para(self):
        '''
        initialize parameters
        '''
        if self.init_rand:
            self.B = []
            self.u = []
            for k in range(self.n_task):
                self.B.append(np.random.random([self.n_feature, self.n_time]))
                self.u.append(np.zeros([self.n_instance[k], self.n_time]))
            self.M = copy.deepcopy(self.Y)

        self.XtX = []
        self.SY = []
        for k in range(self.n_task):
            self.XtX.append(self.X[k].T.dot(self.X[k]))
            self.SY.append(self.S[k]*self.Y[k])

    def get_para(self):
        return self.B, self.M, self.u

    def model_iter(self):
        cnt = 0
        obj_new = 1e16
        obj_old = 1e16
        while cnt < self.max_iter:
            print 'Iteration:',
            print cnt,
            self.update_M()
            self.update_B()
            self.update_u()
            obj_old = obj_new
            obj_new = self.cal_obj(cnt)
            if obj_old - obj_new < 1e-2:
                break
            cnt += 1
            print 'Objective function',
            print obj_new
        print 'loop end'

    def update_M(self):
        for k in range(self.n_task):
            XB = self.X[k].dot(self.B[k])
            self.M[k] = (self.SY[k]+self.rhoADMM*(XB - self.u[k])) / \
                (self.S[k]+self.rhoADMM)
        for k in range(self.n_task):
            self.M[k] = TopUp(self.M[k])

    def update_u(self):
        for k in range(self.n_task):
            self.u[k] += self.M[k] - self.X[k].dot(self.B[k])

    def update_B(self):
        '''
        update B, LF1
        '''
        t_old = 1.0
        t_new = 1.0
        gamma = 1.0
        gamma_inc = 2.0
        self.B_old = copy.deepcopy(self.B)
        n_flag = 0
        cnt = 1
        while cnt < 1000:
            alpha = 0.5
            Bss = []
            for k in range(self.n_task):
                Bss.append((1.0+alpha)*self.B[k] - alpha*self.B_old[k])
            gradB = []
            for k in range(self.n_task):
                gradB.append(update_gradB(self.XtX[k], self.X[k], self.M[k], Bss[k],
                                          self.u[k], self.rhoL2/self.rhoADMM, self.rhoTime/self.rhoADMM))
            funcBs = 0
            for k in range(self.n_task):
                funcBs += update_funcB(self.X[k], Bss[k], self.M[k], self.u[k],
                                       self.rhoL2/self.rhoADMM, self.rhoTime/self.rhoADMM)

            while True:
                VV = []
                for k in range(self.n_task):
                    VV.append(Bss[k]-gradB[k]/gamma)
                Bpp = LF1Projection(VV, self.rhoLF1/gamma/self.rhoADMM)

                funcBp = 0
                for k in range(self.n_task):
                    funcBp += update_funcB(self.X[k], Bpp[k], self.M[k], self.u[k],
                                           self.rhoL2/self.rhoADMM, self.rhoTime/self.rhoADMM)

                delta_B = []
                for k in range(self.n_task):
                    delta_B.append(Bpp[k] - Bss[k])

                r_sum = 0
                for k in range(self.n_task):
                    r_sum += np.linalg.norm(delta_B[k])**2

                funcBp_gamma = funcBs
                for k in range(self.n_task):
                    funcBp_gamma += np.sum(delta_B[k]*gradB[k]) + \
                        0.5*gamma*np.linalg.norm(delta_B[k])**2

                if r_sum < 1e-10:
                    n_flag = 1
                    break

                if funcBp < funcBp_gamma:
                    break

                gamma *= gamma_inc

            if n_flag:
                break

            cnt += 1

            self.B_old = copy.deepcopy(self.B)
            self.B = copy.deepcopy(Bpp)

            t_old = t_new
            t_new = (1.0+np.sqrt(1.0+4.0*t_new**2))*0.5

        print 'B step:',
        print cnt,

    def cal_obj(self, n):
        '''
        Objective Function
        '''
        obj_all = 0.0
        for k in range(self.n_task):
            obj_all += 0.5*np.linalg.norm(self.S[k]*(self.Y[k]-self.M[k]))**2
            obj_all += 0.5*self.rhoADMM * \
                np.linalg.norm(self.X[k].dot(self.B[k])-self.M[k]-self.u[k])**2
            obj_all -= 0.5*self.rhoADMM*np.linalg.norm(self.u[k])**2
            obj_all += 0.5*self.rhoL2*np.linalg.norm(self.B[k])**2
            for j in range(self.n_time-1):
                obj_all += 0.5*self.rhoTime * \
                    np.linalg.norm(self.B[k][:, j]-self.B[k][:, j+1])
        for f in range(self.n_feature):
            num = 0.0
            for k in range(self.n_task):
                num += np.linalg.norm(self.B[k][f])**2
            num = np.sqrt(num)
            obj_all += self.rhoLF1*num
        if n > 0:
            self.obj.append([n, obj_all])
        return obj_all

#     def loss_plot(self):
#         loss_plot(self.obj)

    def save_para(self, files_):
        for k in range(len(self.B)):
            print 'write ', files_[k]
            np.savetxt(files_[k], self.B[k])


if __name__ == '__main__':

    X1 = np.random.random([500, 100])
    X2 = np.random.random([400, 100])
    X3 = np.random.random([700, 100])
    X = [X1, X2, X3]

    Y1 = np.random.random([500, 14])
    for i in range(Y1.shape[0]):
        num = np.random.randint(14)
        for j in range(num):
            Y1[i, j] = 0.0
    Y1[Y1 > 0] = 1.0
    Y2 = np.random.random([400, 14])
    for i in range(Y2.shape[0]):
        num = np.random.randint(14)
        for j in range(num):
            Y2[i, j] = 0.0
    Y2[Y2 > 0] = 1.0
    Y3 = np.random.random([700, 14])
    for i in range(Y3.shape[0]):
        num = np.random.randint(14)
        for j in range(num):
            Y3[i, j] = 0.0
    Y3[Y3 > 0] = 1.0
    Y = [Y1, Y2, Y3]

    S1 = np.ones([500, 14])
    S2 = np.ones([400, 14])
    S3 = np.ones([700, 14])
    S = [S1, S2, S3]

    model = mtLF1(X, Y, S)
    model.init_para()
    model.model_iter()
#     model.loss_plot()

    print 'Normal Terminate'
