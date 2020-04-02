'''
Model Training and Testing.
'''
from model import mtLF1
import pandas
import numpy as np
import os
import csv


def cindex_noncox(predict, time, status):
    pre = 0
    total = 0
    for i in range(0, len(time)-1):
        for j in range(i+1, len(time)):
            if (time[i] < time[j] and predict[i] < predict[j] and status[i] == 1):
                pre += 1
            if (time[i] > time[j] and predict[i] > predict[j] and status[j] == 1):
                pre += 1
            if (time[i] < time[j] and status[i] == 1):
                total += 1
            if (time[i] > time[j] and status[j] == 1):
                total += 1
    if total > 0:
        cindex = float(pre)/float(total)
    else:
        cindex = 0.0
    return cindex


def model_eval(pred_file, test_file):

    beta = np.loadtxt(pred_file, dtype=np.float)

    fp = open(test_file, 'r')
    rd = csv.reader(fp, delimiter=',')
    for line in rd:
        break
    test0 = []
    for line in rd:
        test0.append(line)
    fp.close()

    test_mat = np.zeros([len(test0), beta.shape[0]])
    for i in range(test_mat.shape[0]):
        for j in range(test_mat.shape[1]):
            test_mat[i, j] = test0[i][j]

    pre = test_mat.dot(beta)
    pre = 1.0/(1.0+np.exp(-pre))

    for k in range(pre.shape[0]):
        for j in range(pre.shape[1]-1):
            if pre[k, j+1] > pre[k, j]:
                pre[k, j+1] = pre[k, j]

    pred1 = []
    for k in range(pre.shape[0]):
        mn = np.mean(pre[k][1:pre.shape[1]-1])
        pred1.append(mn)

    test1 = []
    stst1 = []
    for k in range(len(test0)):
        test1.append(int(test0[k][-2]))
        stst1.append(int(test0[k][-1]))
    return cindex_noncox(pred1, test1, stst1)


def model_test(files_coll=[],
               files_pred=[],
               files_test=[],
               fold_n=0,
               max_date=40,
               n_task=4,
               rhoADMM=1.0,
               rhoL2=1.0,
               rhoLF1=10.0,
               rhoTime=0.1,
               max_iter=1):

    X = []
    Z = []
    for file_ in files_coll:
        data = pandas.read_csv(file_)
        data = data.values
        X.append(data[:, :-2])
        Z.append(data[:, -2:])

    Y = []
    S = []
    for mat in Z:
        n_pat = len(mat)
        print n_pat
        n_time = max_date
        yy_ = np.zeros([n_pat, n_time+1])
        ss_ = np.ones([n_pat, n_time+1])
        for kk in range(n_pat):
            yy_[kk, int(mat[kk][0]):] = mat[kk][1]
            ss_[kk, int(mat[kk][0]):] = mat[kk][1]
        yy_ = 1.0 - yy_
        Y.append(yy_)
        S.append(ss_)

    model = mtLF1(X, Y, S, rhoL2=rhoL2, rhoADMM=rhoADMM,
                  rhoLF1=rhoLF1, rhoTime=rhoTime, max_iter=max_iter)
    model.init_para()
    model.model_iter()
    model.save_para(files_=files_pred)

    c_index = []
    for k in range(n_task):
        c_index.append(model_eval(files_pred[k], files_test[k]))

    return c_index


def main():
    rhoLF1 = 5.0
    max_date = 40
    n_task = 4
    rhoADMM = 0.5
    rhoL2 = 0.5
    rhoTime = 2.0
    max_iter = 100
    cin_arr = []
    tmp_folder = 'results'
    if not os.access(tmp_folder, os.F_OK):
        os.mkdir(tmp_folder)
    for fold_n in range(5):
        files_coll = ['data/company_train_'+str(fold_n)+'.csv',
                        'data/manager_train_'+str(fold_n)+'.csv',
                        'data/promotion_train_'+str(fold_n)+'.csv',
                        'data/role_train_'+str(fold_n)+'.csv']

        files_pred = [tmp_folder+'/company_'+str(rhoADMM)+'_'+str(rhoL2)+'_'+str(rhoLF1)+'_'+str(rhoTime)+'_'+str(max_iter)+'_'+str(fold_n)+'_B.csv',
                        tmp_folder+'/manager_'+str(rhoADMM)+'_'+str(rhoL2)+'_'+str(
            rhoLF1)+'_'+str(rhoTime)+'_'+str(max_iter)+'_'+str(fold_n)+'_B.csv',
            tmp_folder+'/promotion_'+str(rhoADMM)+'_'+str(rhoL2)+'_'+str(
            rhoLF1)+'_'+str(rhoTime)+'_'+str(max_iter)+'_'+str(fold_n)+'_B.csv',
            tmp_folder+'/role_'+str(rhoADMM)+'_'+str(rhoL2)+'_'+str(rhoLF1)+'_'+str(rhoTime)+'_'+str(max_iter)+'_'+str(fold_n)+'_B.csv']

        files_test = ['data/company_test_'+str(fold_n)+'.csv',
                        'data/manager_test_'+str(fold_n)+'.csv',
                        'data/promotion_test_'+str(fold_n)+'.csv',
                        'data/role_test_'+str(fold_n)+'.csv']

        cc = model_test(files_coll=files_coll,
                        files_pred=files_pred,
                        files_test=files_test,
                        fold_n=fold_n,
                        max_date=max_date,
                        n_task=n_task,
                        rhoADMM=rhoADMM,
                        rhoL2=rhoL2,
                        rhoLF1=rhoLF1,
                        rhoTime=rhoTime,
                        max_iter=max_iter)
        print cc
        cin_arr.append(cc)
    cin_arr = np.array(cin_arr)
    file_cin = 'cindex_'+str(rhoADMM)+'_'+str(rhoL2)+'_'+str(rhoLF1) + \
        '_'+str(rhoTime)+'_'+str(max_iter)+'_'+str(fold_n)+'.csv'
    np.savetxt(file_cin, cin_arr)


if __name__ == '__main__':
    main()
    print 'Normal Terminate'
