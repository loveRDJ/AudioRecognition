from pydub import AudioSegment
import scipy.io.wavfile as wav
from python_speech_features import mfcc
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
import wave


# def get_ms_part_wav(main_wav_path, start_time, end_time, part_wav_path):
#     start_time = int(start_time)
#     end_time = int(end_time)
#     sound = AudioSegment.from_wav(main_wav_path)
#     output = sound[start_time: end_time]
#     output.export(part_wav_path, format="wav")

def sgn(data):
    if data >= 0:
        return 1
    else:
        return 0

def calculateEnergy(wave_data):
    energy = []
    sum = 0
    for i in range(len(wave_data)):
        sum += (wave_data[i]*wave_data[i])
        # print(sum)
        if (i+1)%256 == 0:
            energy.append(sum)
            sum = 0
        elif i == len(wave_data)-1:
            energy.append(sum)
    # print(energy,"energy")
    return energy

def calculateZCR(wave_data):
    zcr = []
    sum = 0
    for i in range(len(wave_data)):
        if i%256 == 0:
            continue
        sum += np.abs(sgn(wave_data[i]) - sgn(wave_data[i-1]))
        if(i+1)%256 == 0:
            zcr.append(float(sum/255))
            sum = 0
        elif i == len(wave_data) - 1:
            zcr.append(float(sum/255))
    # print(zcr,"zcr")
    return zcr

def endPointDetect(wave_data, energy, zcr):
    sum = 0
    avgEnergy = 0
    for e in energy:
        sum += e
    avgEnergy = sum/len(energy)

    sum = 0
    for e in energy[:5]:
        sum += e
    ML = sum/5
    MH = avgEnergy/4
    ML = (ML+MH)/4

    sum = 0
    for z in zcr[:5]:
        sum += float(sum+z)
    Zs = sum/5

    A = []
    B = []
    C = []

    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i]>MH:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i]>MH and i-21>A[len(A)-1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i]>MH and i-21<=A[len(A)-1]:
            A = A[:len(A)-1]
            flag = 1

        if flag == 1 and energy[i]<MH:
            A.append(i)
            flag = 0

    for j in range(len(A)):
        i = A[j]
        if j%2 ==1:
            while i< len(energy) and energy[i]>ML:
                i += 1
            B.append(i)
        else:
            while i>0 and energy[i]>ML:
                i -= 1
            B.append(i)

    for j in range(len(B)):
        i = B[j]
        if j%2 == 1:
            while i<len(zcr) and zcr[i]>=3*Zs:
                i += 1
            C.append(i)
        else:
            while i>0 and zcr[i]>=3*Zs:
                i -= 1
            C.append(i)

    return C


def get_mfcc(wav_path):
    rate, audio = wav.read(wav_path)
    # 语音有效段检测-短时平均过零率
    f = wave.open(wav_path, "rb")
    params = f.getparams()
    # 读取格式信息
    # (声道数、量化位数、采样频率、采样点数、压缩类型、压缩类型的描述)
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()

    # 将字符串转换为数组，得到一维的short类型的数组
    wave_data = np.fromstring(str_data, dtype=np.short)
    # 赋值的归一化
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))
    # 整合左声道和右声道的数据
    wave_data = np.reshape(wave_data, [nframes, nchannels])
    # wave_data.shape = (-1, 2)
    # print(wave_data)
    energy = calculateEnergy(wave_data)
    zcr = calculateZCR(wave_data)
    N = endPointDetect(wave_data, energy, zcr)
    print(N[0],"N")
    # fft_signal = np.fft.fft(wave_data)  # 语音信号FFT变换
    # fft_signal = abs(fft_signal)  # 取变换结果的模
    # plt.figure(figsize=(10, 4))
    # time = np.arange(0, nframes) * framerate / nframes
    # plt.plot(time, fft_signal, c="g")
    # plt.grid()
    # plt.show()

    return mfcc(audio, rate)


if __name__ == "__main__":
    # get_ms_part_wav('VR.mp3', 1000, 6000, 'Test_data/VR.wav')
    orig1 = get_mfcc('1.wav')
    print(orig1.shape)
    # plt.plot(orig1)
    # plt.show()
