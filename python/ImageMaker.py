import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os
import librosa.display
audios = librosa.util.find_files(r"./audio_to_graph")
for filename in audios:
    f = wave.open(filename,'rb')
    # 得到语音参数
    params = f.getparams()
    nchannels, sampwidth, framerate,nframes = params[:4]
    # 得到的数据是字符串，需要将其转成int型
    strData = f.readframes(nframes)
    wavaData = np.fromstring(strData,dtype=np.int16)
    # 归一化
    wavaData = wavaData * 1.0/max(abs(wavaData))
    # .T 表示转置
    wavaData = np.reshape(wavaData,[nframes,nchannels]).T
    f.close()
    # 绘制频谱
    plt.specgram(wavaData[0],Fs = framerate,scale_by_freq=True,sides='default')
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.savefig("test"+str(audios.index(filename))+".png")