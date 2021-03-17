import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import mixture

def get_label():
    file_list = [filename for filename in os.listdir(r"./data")]
    return file_list

def MFCC():
    audios = librosa.util.find_files(r"./data")
    mfccs=[]
    for i in range(len(audios)):
        x, sr = librosa.load(audios[i])
        if (len(x) >= 50000):
            x=x[0:50000]
        if (len(x) < 50000):
            x = np.pad(x, (0, 50000 - len(x)), 'constant', constant_values=(0, 0))
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=20)

        mfccs.append(mfcc)
    mfccs=np.array(mfccs)
    print("mfccs_shape={}".format(np.shape(mfccs)))



    return mfccs

def Norm(mfccs):
    C=np.zeros([5,98,20])
    for i in range(len(mfccs)):
        for j in range(len(mfccs[0][0])):
            for m in range(len(mfccs[0])):
                ci=mfccs[i][m][j]
                C[i][j][m]=ci

                # if i == 0:#检查所有Ci都已经标准化
                #     if j == 0:
                #         if m==0:
                #             print(C[i][j])
                transformer = StandardScaler().fit(C[i][j].reshape(-1,1))
                new= transformer.transform(C[i][j].reshape(-1,1))
                C[i][j]=new.reshape(1,-1)
            # if i==0:              #检查所有Ci都已经标准化
            #     if j==0:
            #         print(C[i][j])
    New_mfccs=np.zeros([5,20,98])
    for i in range(len(C)):
        for j in range(len(C[0])):
            for m in range(len(C[0][0])):
                cii=C[i][j][m]
                New_mfccs[i][m][j]=cii

    print("C_shape={},对应(speaker,ci,num_ci)".format(np.shape(C)))
    print("New_mfccs_shape={},对应(speaker,num_ci,ci)".format(np.shape(New_mfccs)))
    return C,New_mfccs
def GMMmodel(C,New_mfccs):
    #取出说话人1 标准化后的c1-c98存入numpy

    Norms1=np.zeros([98,20])
    for i in range(len(C[0])):
        Norms1[i]=C[0][i]
    Norms2 = np.zeros([98, 20])
    for i in range(len(C[0])):
        Norms2[i] = C[1][i]
    Norms3 = np.zeros([98, 20])
    for i in range(len(C[0])):
        Norms3[i] = C[2][i]
    Norms4 = np.zeros([98, 20])
    for i in range(len(C[0])):
        Norms4[i] = C[3][i]
    Norms5 = np.zeros([98, 20])
    for i in range(len(C[0])):
        Norms5[i] = C[4][i]
    print(np.shape(Norms1))
    print('231312312312')
    print(np.shape(Norms2))
    print(np.shape(Norms3))
    model = mixture.GaussianMixture(n_components=5,covariance_type='spherical',max_iter=10,warm_start=True)
    model.fit(Norms1)
    MEAN1=model.means_
    COVAR1=model.covariances_
    model.fit(Norms2)
    MEAN2=model.means_
    COVAR2 = model.covariances_
    model.fit(Norms3)
    MEAN3 = model.means_
    COVAR3 = model.covariances_
    model.fit(Norms4)
    MEAN4 = model.means_
    COVAR4 = model.covariances_
    model.fit(Norms5)
    MEAN5 = model.means_
    COVAR5 = model.covariances_
    print(np.shape(MEAN1))

    COVAR1_norm = np.linalg.norm(COVAR1)
    COVAR2_norm = np.linalg.norm(COVAR2)
    COVAR3_norm = np.linalg.norm(COVAR3)
    COVAR4_norm = np.linalg.norm(COVAR4)
    COVAR5_norm = np.linalg.norm(COVAR5)
    dif12 = np.sqrt(np.sum(np.square(MEAN1 - MEAN2)))
    dif13 = np.sqrt(np.sum(np.square(MEAN1 - MEAN3)))
    dif14 = np.sqrt(np.sum(np.square(MEAN1 - MEAN4)))
    dif15 = np.sqrt(np.sum(np.square(MEAN1 - MEAN5)))
    print("MEAN")
    print(dif12)
    print(dif13)
    print(dif14)
    print(dif15)


    cos12 = np.dot(COVAR1, COVAR2) / (COVAR1_norm * COVAR2_norm)
    cos13 = np.dot(COVAR1, COVAR3) / (COVAR1_norm * COVAR3_norm)
    cos14 = np.dot(COVAR1, COVAR4) / (COVAR1_norm * COVAR4_norm)
    cos15 = np.dot(COVAR1, COVAR5) / (COVAR1_norm * COVAR5_norm)
    print("COS")
    print(cos12)
    print(cos13)
    print(cos14)
    print(cos15)



def main():
    label=get_label()
    mfccs=MFCC()        #提取特征mfccs
    C,New_mfccs=Norm(mfccs)  #标准化mfccs
                        #取一阶，二阶导为特征
    GMMmodel(C,New_mfccs)          #GMM建模

if __name__ == '__main__':
    main()