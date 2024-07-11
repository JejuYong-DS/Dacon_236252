# -*- coding: utf-8 -*-
"""
SW중심대학 디지털 경진대회
데이터 분석 파이프라인

"""
#%%
# !pip install pydub
# !pip install webrtcvad
#%% module
import numpy as np
import pandas as pd
import os

import librosa
import webrtcvad

#%% path
link = r'C:\Users\PC\Desktop\Korea_Univ\CDS_LAB\연구3\소중대_데이콘\data'

#%% Data load
list_unlabeled_name = os.listdir(link+r'./unlabeled_data')
list_unlabeled_name = [r'./unlabeled_data/' + s for s in list_unlabeled_name]

unlabeled_ogg_list = []
for i, audio_path in enumerate(list_unlabeled_name):
    #진행 상황
    print(i/len(list_unlabeled_name))
    
    y, sr = librosa.load(audio_path, sr=32000)
    unlabeled_ogg_list.append(y)

    #실험용 코드
    # if i == 100 : break
#%% 1. 음성 활동 감지
def VAD(audio, sr=32000, vad_mode = 0, ms = 10, min_speech_duration_ms = 2000):
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

list_unlabeled_name[600]
result_step_1_list = []
for i in range(len(list_unlabeled_name)):
    result_step_1 = VAD(unlabeled_ogg_list[i], vad_mode = 0, ms = 10)
    result_step_1_list.append(result_step_1)
    print(i, result_step_1)

print(sum(result_step_1_list))
