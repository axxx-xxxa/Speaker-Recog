import librosa.display
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import mixture
from sklearn.cluster import KMeans
import os
import pandas as pd
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
    print(len(mfccs))
    print(len(mfccs[0]))
    print(len(mfccs[0][0]))
    return mfccs

def Norm_mfccs(mfccs):
    for m in range(len(mfccs[0][0])):
        # 每一个语音数据
        for i in range(len(mfccs)):
             #每一段MFCC
            for j in range(len(mfccs[0])):
                mfccs[i][j][m]




def model(feature,label):
    model = mixture.GaussianMixture(n_components=2,covariance_type='full',max_iter=10)
    # model = KMeans(n_clusters=2)
    # model= SVC()
    #feature(12,40,196)
    feature=np.array(feature)

    ##这一步可能转的有问题！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
    # model=model.fit(feature,label)

def main():
    label=get_label()
    mfccs=MFCC()
    model(mfccs,label)
if __name__ == '__main__':
    main()


