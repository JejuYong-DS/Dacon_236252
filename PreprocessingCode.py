# -*- coding: utf-8 -*-
"""
SW중심대학 디지털 경진대회
Preprocessing

"""
#%% module
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#!pip install librosa
import librosa
import librosa.display

#!pip install catboost
#!pip install xgboost

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

import warnings
warnings.filterwarnings('ignore')
#%% path
link = r'C:\Users\PC\Desktop\Korea_Univ\CDS_LAB\연구3\소중대_데이콘\data'
# 추가

#%% data load
df_train_label = pd.read_csv('train.csv', encoding='utf-8')
mapping = {'fake':0,'real':1}
df_train_label.label = df_train_label.label.map(mapping)

train_ogg_list = []
for i, audio_path in enumerate(df_train_label['path']):    
    #진행 상황
    print(i/len(df_train_label))

    y, sr = librosa.load(audio_path, sr=32000) #16000, 32000
    train_ogg_list.append(y)
    
    #시각화
    # plt.plot(y)
    # plt.title(str(i)+'_'+df_train_label['label'][i])
    # plt.show()
    
    #실험용 코드
    # if i == 100 : df_train_label_100 = df_train_label[:101] ; break

#전처리 코드 완성 후에 풀기 (또는 파이프라인 코드로 옮기기 또는 이걸 전처리 코드 메인으로 삼기)
list_unlabeled_name = os.listdir(link+r'\unlabeled_data')
list_unlabeled_name = [r'\unlabeled_data\\' + s for s in list_unlabeled_name]

unlabeled_ogg_list = []
for i, audio_path in enumerate(list_unlabeled_name):
    print(i/len(list_unlabeled_name))

    y, sr = librosa.load(link + audio_path, sr=32000) #16000, 32000
    unlabeled_ogg_list.append(y)
    # plt.plot(y)
    # plt.title(str(i))
    # plt.show()

#%%
# #%% STFT : Short Time Fourier Transform
# def STFT(y, n_fft = 1024): #512, 1024, 2048
#     D = librosa.stft(y, n_fft=n_fft, win_length = n_fft, hop_length=n_fft//4)
#     S_dB = librosa.amplitude_to_db(abs(D), ref=np.max)
#     librosa.display.specshow(S_dB, sr=sr, 
#                              hop_length = n_fft//4, 
#                              x_axis='time', y_axis='log')
#     plt.colorbar(format='%2.0f dB')
#     plt.show()
#     return S_dB

# train_s_db_list = []
# for i in range(len(train_ogg_list)):
#     s_db = STFT(train_ogg_list[i])
#     train_s_db_list.append(s_db)

#%% MFCC
# n_mfcc = 13
# def MFCC(y, n_mfcc = 20): #13, 20
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
#     print(mfccs)
    
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_dB = librosa.amplitude_to_db(S, ref=np.max)
#     plt.figure(figsize=(14, 5))
#     librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel spectrogram')
#     plt.show()

#%% 음성 데이터에서 특성 추출하기
def STFT_MFCC(i, y, sr = 32000, n_fft = 1024, n_mfcc = 13): #512, 1024, 2048
    #STFT : Short Time Fourier Transform
    D = librosa.stft(y, n_fft=n_fft, win_length = n_fft, hop_length=n_fft//4)
    stft_db = librosa.amplitude_to_db(abs(D), ref=np.max)
    stft_features = np.mean(stft_db, axis=1)

    #MFCC : Mel-Frequency Cepstral Coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=n_fft//4)
    mfcc_features = np.mean(mfcc, axis=1)
    
    #concat
    features = np.concatenate((stft_features, mfcc_features))
    
    #진행 상황
    print(i)
    
    return features

#%% 특성 추출
X = np.array([STFT_MFCC(i, y) for i, y in enumerate(train_ogg_list)])
y = np.array(df_train_label['label'])

#%%
acc_dict = {}
for random_seed in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
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
    
    break
    
    print(f"Accuracy: {accuracy}")
    acc_dict[str(random_seed)] = accuracy

print(acc_dict)

#%% Test data    
df_test_label = pd.read_csv('test.csv', encoding='utf-8')

test_ogg_list = []
for i, audio_path in enumerate(df_test_label['path']):    
    #진행 상황
    print(i/len(df_test_label))

    y, sr = librosa.load(audio_path, sr=32000) #16000, 32000
    test_ogg_list.append(y)


X_test = np.array([STFT_MFCC(i, y) for i, y in enumerate(test_ogg_list)])
X_test = scaler.transform(X_test)


# 예측
y_predict_proba = model.predict_proba(X_test)


df_predict_proba = pd.DataFrame(y_predict_proba, columns = ['fake', 'real'])


df_sample_submission = pd.read_csv('sample_submission.csv', encoding='utf-8')
df_sample_submission = pd.concat((df_sample_submission['id'],df_predict_proba), axis=1)

df_sample_submission





