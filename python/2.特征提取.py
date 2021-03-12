import librosa
import pyaudio
import wave
import numpy as np

y, sr = librosa.load('./audioinput.wav', sr=1600)
# 提取 MFCC feature
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# 对特征数据进行归一化，减去均值除以方差
feature_inputs = np.asarray(mfccs[np.newaxis, :])
feature_inputs = (feature_inputs - np.mean(feature_inputs)) / np.std(feature_inputs)

# 特征数据的序列长度
feature_seq_len = [feature_inputs.shape[1]]

print(mfccs.shape)
print(feature_inputs.shape)
print(feature_seq_len)