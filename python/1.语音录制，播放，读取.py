import pyaudio
import wave
from tqdm import tqdm


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
def record_video(path,seconds):
    #创建播放器
    p = pyaudio.PyAudio()
    #stream开启数据结构为p的数据流(开启线程)
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("start recording......")
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    #数据流存取的次数=（采样频率*时间）/缓冲区大小
    for i in tqdm(range(0, int(RATE / CHUNK * seconds))):
        data = stream.read(CHUNK)
        wf.writeframes(data)

    print("end!")
    #停止流
    stream.stop_stream()
    #关闭流
    stream.close()
    #关闭创建播放器
    p.terminate()

    wf.close()

def play_video(path):
     #打开文件
     wf = wave.open(path,'rb')
     #打开播放器
     p = pyaudio.PyAudio()

     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                     channels=wf.getnchannels(),
                     rate=wf.getframerate(),
                     output=True)
     data = wf.readframes(CHUNK)

     datas=[]
     while len(data)>0:
         data=wf.readframes(CHUNK)
         datas.append(data)

     for d in tqdm(datas):
         stream.write(d)

     stream.stop_stream()
     stream.close()

     p.terminate()

def main():
    record_video("output.wav",1)
    play_video("output.wav")

if __name__ == '__main__':
    main()