# -*- coding: utf-8 -*-
"""
SW중심대학 디지털 경진대회
Preprocessing

"""
#%% module
# !pip install librosa
# !pip install xgboost
# !pip install pyannote.audio
# !pip install torch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

from scipy.fft import fft, ifft
import webrtcvad

import torch
from pyannote.audio import Pipeline

import warnings
warnings.filterwarnings('ignore')
#%% path
link = r'C:\Users\PC\Desktop\Korea_Univ\CDS_LAB\연구3\소중대_데이콘\data'
api_key = ""


pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=api_key)

#%% data load
df_train_label = pd.read_csv('train.csv', encoding='utf-8')
mapping = {'fake':0,'real':1}
df_train_label.label = df_train_label.label.map(mapping)

train_ogg_list = []
for i, audio_path in enumerate(df_train_label['path']):    
    #진행 상황
    print(i/len(df_train_label))

    y, sr = librosa.load(audio_path, sr=32000)
    train_ogg_list.append(y)

#%% 음성 데이터에서 특성 추출하기
def STFT_MFCC(y, i=None, sr = 32000, n_fft = 1024, n_mfcc = 13): #512, 1024, 2048
    # STFT : Short Time Fourier Transform
    D = librosa.stft(y, n_fft=n_fft, win_length = n_fft, hop_length=n_fft//4)
    stft_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    stft_features = np.mean(stft_db, axis=1)

    #MFCC : Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft//4)
    mfcc_features = np.mean(mfcc, axis=1)
    
    #concat
    features = np.concatenate((stft_features, mfcc_features))
    #진행 상황
    if i :
        print(i)
    
    return features

#%% 특성 추출
X = np.array([STFT_MFCC(y=audio, i=i) for i, audio in enumerate(train_ogg_list)])
y = np.array(df_train_label['label'])

#%%
# acc_dict = {}
# for random_seed in range(100):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #random_seed

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoostClassifier 모델 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)
y_predict_proba = model.predict_proba(X_test)
# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
    # acc_dict[str(random_seed)] = accuracy
    # break

# print(acc_dict)

#%%
def visualization(audio, cleaned_audio):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(audio)
    plt.title('Original Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(cleaned_audio)
    plt.title('Cleaned Audio (Noise Removed)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.title(str(i))
    plt.show()

def noise_removal(audio, noise_freq_band=(2000, 4000)):
    # audio = audio.numpy()
    
    # FFT를 이용한 주파수 스펙트럼 분석
    n = len(audio)
    audio_fft = fft(audio)
    freq = np.fft.fftfreq(n, d=1/32000)  # 주파수 축 계산

    # 노이즈 주파수 대역 설정
    noise_mask = np.logical_or(freq < noise_freq_band[0], freq > noise_freq_band[1])

    # 노이즈 대역을 제외한 주파수 스펙트럼 계산
    clean_fft = audio_fft.copy()
    clean_fft[noise_mask] = 0  # 노이즈 대역의 주파수 성분을 0으로 설정
    
    # IFFT를 이용한 필터링된 시간 영역 신호 복원
    cleaned_audio = np.real(ifft(clean_fft))
    
    # visualization(audio, cleaned_audio)

    return cleaned_audio


#%% Test data load
df_test_label = pd.read_csv('test.csv', encoding='utf-8')

test_ogg_list = []
for i, audio_path in enumerate(df_test_label['path']):    
    #진행 상황
    print(i/len(df_test_label))

    y, sr = librosa.load(audio_path, sr=32000) #32000
    y = noise_removal(y) # 노이즈 제거
    test_ogg_list.append(y)
    
#%% 1. 음성 활동 감지
def VAD(audio, sr=32000, vad_mode = 0, ms = 10, min_speech_duration_ms = 300):
    # 16-bit PCM으로 변환
    samples = (audio * 32767).astype(np.int16).tobytes()
    
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)  # Set aggressiveness mode (0-3): higher values are more aggressive
    frame_size = int(sr * ms / 1000 * 2) # ms : (10-30)
    
    is_speech = False
    min_frames_for_speech = int(min_speech_duration_ms / ms)
    consecutive_speech_frames = 0
    for i in range(0, len(samples), frame_size):
        frame = samples[i:i+frame_size]
        if len(frame) < frame_size:
            break
        
        if vad.is_speech(frame, sr):
            consecutive_speech_frames += 1
        else:
            consecutive_speech_frames = 0
        
        if consecutive_speech_frames >= min_frames_for_speech:
            is_speech = True
            break
    
    return is_speech
#%% VAD use pyannote
# audio = unlabeled_ogg_list[3]
# from pyannote.audio import Pipeline, Model

# model = Model.from_pretrained("pyannote/segmentation", 
#                                     use_auth_token=api_key
#                                     )
# from pyannote.audio.pipelines import VoiceActivityDetection
# pipeline = VoiceActivityDetection(segmentation=model)
# HYPER_PARAMETERS = {
#   # onset/offset activation thresholds
#   "onset": 0.75, "offset": 0.37,
#   # remove speech regions shorter than that many seconds.
#   "min_duration_on": 0.13,
#   # fill non-speech regions shorter than that many seconds.
#   "min_duration_off": 0.067
# }

# pipeline.instantiate(HYPER_PARAMETERS)
# vad = pipeline({"waveform":audio, "sample_rate": 32000})


# from pyannote.audio.pipelines import OverlappedSpeechDetection
# pipeline = OverlappedSpeechDetection(segmentation=model)
# pipeline.instantiate(HYPER_PARAMETERS)
# osd = pipeline({"waveform":audio, "sample_rate": 32000})

# from pyannote.audio.pipelines import Resegmentation
# pipeline = Resegmentation(segmentation=model, 
#                           diarization="baseline")
# pipeline.instantiate(HYPER_PARAMETERS)
# resegmented_baseline = pipeline({"audio": "audio.wav"})


# def VAD(audio):
#     speech_segments = []
#     for segment in vad.get_timeline().support():
#         start = int(segment.start * sr)
#         end = int(segment.end * sr)
#         speech_segments.append(audio[start:end])

#%% 2. 화자 탐지
audio = unlabeled_ogg_list[3]
def speaker_diarization(audio):
    audio = torch.Tensor([audio])
    # apply pretrained pipeline
    diarization = pipeline({"waveform": audio, "sample_rate": 32000}, min_speakers=1, max_speakers=2)

    speakers = {}
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * 32000) ; end = int(turn.end * 32000)
        speaker_audio = audio[0][start:end]
        speakers[f'{speaker}'] = speaker_audio
    print(speakers)
    if len(speakers.keys()) == 2:
        return STFT_MFCC(y=speakers['SPEAKER_00'])
    
    else :
        X_test = STFT_MFCC(y=speakers['SPEAKER_00'].numpy())
        X_test = scaler.transform(X_test)
        X_test_predict_proba = model.predict_proba(X_test)
        
#%% CDS_Model
def CDS_Model(audio):
    result_step_1 = VAD(audio, vad_mode = 0, ms = 10)
    
    if not result_step_1:
        return [0,0]
    
    result = speaker_diarization(audio)
    
    
    
#%%
X_test = np.array([STFT_MFCC(y=audio, i=i) for i, audio in enumerate(test_ogg_list)])
X_test = scaler.transform(X_test)

# 예측
y_predict_proba = model.predict_proba(X_test)

df_predict_proba = pd.DataFrame(y_predict_proba, columns = ['fake', 'real'])

df_sample_submission = pd.read_csv('sample_submission.csv', encoding='utf-8')
df_sample_submission = pd.concat((df_sample_submission['id'],df_predict_proba), axis=1)

df_sample_submission







