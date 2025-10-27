# -*- coding: utf-8 -*-
"""
SW중심대학 디지털 경진대회
Preprocessing

"""
#%% module
# !pip install librosa xgboost pyannote.audio torch webrtcvad
# !pip install webrtcvad
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
link = r'C:\Users\rhwnd\Downloads\open' # 개인 PC 환경에 맞추어 수정
api_key = ""
os.chdir(link)

pipeline_voice_activity_detection = Pipeline.from_pretrained(
    "pyannote/voice-activity-detection",
    use_auth_token=api_key)

pipeline_speaker_diarization = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=api_key)

#%% data load
df_train_label = pd.read_csv('train.csv', encoding='utf-8')
mapping = {'fake':0,'real':1}
df_train_label.label = df_train_label.label.map(mapping)

train_ogg_list = []
for i, audio_path in enumerate(df_train_label['path']):    
    #진행 상황
    if i % (len(df_train_label) // 10) == 0: print(f'{i/len(df_train_label)*100}%')

    y, sr = librosa.load(audio_path, sr=32000)
    train_ogg_list.append(y)

print('---train_data_load_complete---')
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

print('---feature_extraction_complete---')

#%%
# acc_dict = {}
# for random_seed in range(100):
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0) #random_seed

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# XGBoostClassifier 모델 학습
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_val)
# 모델 평가
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f"Accuracy: {accuracy}")
print(report)
    # acc_dict[str(random_seed)] = accuracy
    # break

# print(acc_dict)
print('---modeling_complete---')
#%% 각 관측치마다 predict_proba를 구하는 함수
def model_test(test_data):
    test_data = scaler.transform(test_data)
    return model.predict_proba(test_data)
#%%
def visualization(audio, cleaned_audio):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(audio)
    plt.title('Original Audio')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.subplot(1, 2, 2)
    plt.plot(cleaned_audio)
    plt.title('Cleaned Audio (Noise Removed)')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.title(str(i))
    plt.show()
#%%
def noise_removal(audio, noise_freq_band=(1000, 2000)):
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
#%%
def data_cleaning(audio):
    # Convert the audio to a PyTorch tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
    
    # Apply the VAD pipeline
    vad_scores = pipeline_voice_activity_detection({"waveform": audio_tensor, "sample_rate": 32000})
    
    # Create an empty array to store the cleaned audio
    cleaned_audio = np.zeros_like(audio)
    
    # Extract speech segments based on VAD results
    for segment in vad_scores.get_timeline().support():
        start = int(segment.start * sr)
        end = int(segment.end * sr)
        cleaned_audio[start:end] = audio[start:end]
    return cleaned_audio
    # visualization(audio, cleaned_audio)
    
#%% Test data load
df_test_label = pd.read_csv('test.csv', encoding='utf-8')

test_ogg_list = []
for i, audio_path in enumerate(df_test_label['path']):    
    #진행 상황
    if i % (len(df_test_label) // 10) == 0: print(f'{i/len(df_test_label)*100}%')

    y, sr = librosa.load(audio_path, sr=32000) #32000
    y = data_cleaning(y) # 노이즈 제거
    test_ogg_list.append(y)
    
print('---test_data_load_complete---')

#%% 모델링
def CDS_Model(audio):
    audio = torch.Tensor([audio])
    # apply pretrained pipeline
    diarization = pipeline_speaker_diarization({"waveform": audio, "sample_rate": 32000}, min_speakers=0, max_speakers=2)
    
    speakers = {}
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = int(turn.start * 32000) ; end = int(turn.end * 32000)
        speaker_audio = audio[0][start:end]
        speakers[f'{speaker}'] = speaker_audio
    # print(len(speakers.keys()))
    
    if len(speakers.keys()) == 2:
        test_data_0 = STFT_MFCC(y=speakers['SPEAKER_00'].numpy()).reshape(1,-1)
        predict_proba_0 = model_test(test_data_0)[0]
        test_data_1 = STFT_MFCC(y=speakers['SPEAKER_01'].numpy()).reshape(1,-1)
        predict_proba_1 = model_test(test_data_1)[0]

        # 분류 확률이 .40 ~ .60 사이일 경우, 소음으로 판단.
        if max(predict_proba_0) <= 0.60 : return predict_proba_1.tolist()
        if max(predict_proba_1) <= 0.60 : return predict_proba_0.tolist()
        
        # 두 목소리가 real & fake일 경우, [1,1]로 return
        if (predict_proba_0[0] > predict_proba_0[1]) != (predict_proba_1[0] > predict_proba_1[1]) : return [1,1]
        
        # 두 목소라가 real,real이거나 fake,fake일 경우, 최대 확률로 return
        if max(predict_proba_0) > max(predict_proba_1): return predict_proba_0.tolist()
        else : return predict_proba_1.tolist()

    elif len(speakers.keys()) == 1:
        test_data = STFT_MFCC(y=speakers['SPEAKER_00'].numpy()).reshape(1,-1)
        return model_test(test_data)[0].tolist()
    else :
        return [0,0]

#%% 예측
list_predict_proba = []
for i,a in enumerate(test_ogg_list):
    #진행 상황
    if i % (len(test_ogg_list) // 1000) == 0: print(f'{i/len(test_ogg_list)*100}%')
    list_predict_proba.append(CDS_Model(a))

# 예측결과 저장    
df_predict_proba = pd.DataFrame(list_predict_proba, columns = ['fake', 'real'])
df_sample_submission = pd.read_csv('sample_submission.csv', encoding='utf-8')
df_sample_submission = pd.concat((df_sample_submission['id'],df_predict_proba), axis=1)

# csv 파일 저장
df_sample_submission.to_csv('sample_submission.csv', encoding = 'utf-8', index=False)

print('predict_proba_complete')

