# SW_Dacon

"SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI 부문" 데이콘 공모전

CDS팀

total.py 파일을 다운받으시면 됩니다.

<분석 과정>

음성데이터에서 STFT(Short Time Fourier Transform)와 MFCC(Mel-Frequency Cepstral Coefficient)를 통해 특성 추출

XGBoost로 Classification 진행

pyannote API를 사용하여 음성 탐지 및 화자 분리 진행

화자 분리 결과에 따라 분리하여 예측 진행(0인, 1인, 2인)

결과
- 0인 : [0,0]
- 1인 : [P(fake), P(real)]
- 2인(한 명은 fake, 한 명은 real인 경우) : [1,1]
- 2인 : [P_1(fake), P_1(real)]과 [P_2(fake), P_2(real)] 중 |P(fake) - P(real)|가 더 큰 결과를 선택
