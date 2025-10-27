# -*- coding: utf-8 -*-
"""
SW중심대학 디지털 경진대회
데이터 분석 파이프라인

"""
#%%
# !pip install webrtcvad
# !pip install uisrnn

#%% module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

import librosa
from scipy.fft import fft, ifft
import webrtcvad
import uisrnn

import torch
import torchaudio

#%% path
link = r'C:\Users\PC\Desktop\Korea_Univ\CDS_LAB\연구3\소중대_데이콘\data'

api_key = ""

#%% 노이즈 제거
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
    audio = audio.numpy()
    
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

def noise_removal_stft(audio):
    spectrogram = np.abs(librosa.stft(audio))
    
    noise_mask = np.mean(spectrogram, axis = 1)
    
    spectrogram[noise_mask] = 0
    
    cleaned_audio = librosa.istft(spectrogram)
    visualization(audio, cleaned_audio)
    return cleaned_audio

#%% Data load
list_unlabeled_name = os.listdir(link+r'./unlabeled_data')
list_unlabeled_name = [r'./unlabeled_data/' + s for s in list_unlabeled_name]
list_unlabeled_name[100]
unlabeled_ogg_list = []
for i, audio_path in enumerate(list_unlabeled_name):
    #진행 상황
    print(i/len(list_unlabeled_name))
    
    y, sr = librosa.load(audio_path, sr = 32000)
    
    # y = noise_removal(y)
    
    unlabeled_ogg_list.append(y)
    
    #실험용 코드
    if i == 100 : break

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

result_step_1_list = []
for i in range(len(unlabeled_ogg_list)):
    result_step_1 = VAD(unlabeled_ogg_list[i], vad_mode = 0, ms = 10)
    result_step_1_list.append(result_step_1)

    if not result_step_1:
        print(i, result_step_1)
sum(result_step_1_list)

#%% 1. 음성 활동 감지
def VAD(audio, sr=32000, vad_mode = 0, ms = 10):
    # 16-bit PCM으로 변환
    samples = (audio * 32767).astype(np.int16).tobytes()
    
    vad = webrtcvad.Vad()
    vad.set_mode(vad_mode)  # Set aggressiveness mode (0-3): higher values are more aggressive
    frame_size = int(sr * ms / 1000 * 2) # ms : (10-30)
    
    is_speech = False
    for i in range(0, len(samples), frame_size):  # 2 bytes per sample for 16-bit audio
        frame = samples[i:i+frame_size]
        if len(frame) < frame_size:
            break
        if vad.is_speech(frame, sr):
            is_speech = True
            break

    return is_speech

result_step_1_list = []
for i in range(len(unlabeled_ogg_list)):
    result_step_1 = VAD(unlabeled_ogg_list[i], vad_mode = 0, ms = 10)
    result_step_1_list.append(result_step_1)
    print(i, result_step_1)

sum(result_step_1_list)
    
#%% 2. 다중 화자 감지
# def step_2():
#     result = 1,2
#     return result
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
y = unlabeled_ogg_list[0]
for y in unlabeled_ogg_list:
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Transpose MFCCs for clustering (frames, features)
    mfccs = mfccs.T
    
    # Define number of clusters (e.g., 2 for 1 or 2 speakers)
    num_clusters = 2
    
    # Apply clustering (e.g., K-means)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(mfccs)
    
    # Count unique cluster labels to determine number of speakers
    num_speakers = len(np.unique(cluster_labels))
    
    # Print the number of speakers detected
    print(f"Number of speakers detected: {num_speakers}")
    
#%%
# !pip install pyannote.audio
# !pip install torch
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=api_key)

# GPU 사용으로 전환
import torch
pipeline.to(torch.device("cuda"))

#%%
y = unlabeled_ogg_list[0]
for y in unlabeled_ogg_list:
    y = torch.Tensor([y])
    # apply pretrained pipeline
    diarization = pipeline({"waveform": y, "sample_rate": sr}, min_speakers=0, max_speakers=2)
    
    speakers = set()
    # print the result
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        
        speakers.add(speaker)
    print(speakers)

#%%
y = train_ogg_list[0]
y = unlabeled_ogg_list[0]

frame_length = 1024
energy = np.array([sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), frame_length)])

# Calculate ratio of high energy frames
threshold_energy = 0.3  # Adjust this threshold as needed
high_energy_ratio = np.sum(energy > threshold_energy * np.max(energy)) / len(energy)
print(high_energy_ratio)
# Decision based on the ratio
if high_energy_ratio > 0.5:
    print("1 speaker detected")
else:
    print("More than 1 speaker detected")
    
    
    
#%% 3. 화자 분리 : 음원 분리 기법
from sklearn.mixture import GaussianMixture

def step_3(y):
    mfcc = librosa.feature.mfcc(y=y, sr=32000, n_mfcc=13, n_fft=1024, hop_length=1024//4)

    # 군집의 수(화자 수)
    num_speakers = 2
    
    # Transpose MFCC matrix
    mfccs_transposed = mfcc.T
    
    # Gaussian Mixture Model 초기화
    gmm = GaussianMixture(n_components=num_speakers, covariance_type='diag', random_state=0)
    gmm.fit(mfccs_transposed)
    
    # 화자를 예측하여 라벨링
    speaker_labels = gmm.predict(mfccs_transposed)

    # 화자 분리
    speaker_0 = mfcc[:, speaker_labels == 0]
    speaker_1 = mfcc[:, speaker_labels == 1]

    return speaker_0, speaker_1

#%% 분류 모형 모델링
def determine():
    if VAD():
        return [0,0]
    if step_2() == 2:
        step_3()
        
        
















