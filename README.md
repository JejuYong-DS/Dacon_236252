# SW_Dacon

"SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI 부문" 데이콘 공모전

CDS팀

음성데이터에서 STFT(Short Time Fourier Transform)와 MFCC(Mel-Frequency Cepstral Coefficient)를 통해 특성 추출

XGBoost로 Classification 진행

pyannote API를 사용하여 음성 탐지 및 화자 분리 진행

화자 분리 결과에 따라 분리하여 예측 진행(0인, 1인, 2인)

결과는 [P(fake), P(real)]
