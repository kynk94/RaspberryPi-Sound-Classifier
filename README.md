# 2019년 연세대학교 소셜 챌린지 메이커톤
<img src="https://user-images.githubusercontent.com/41245985/62367294-2db98200-b564-11e9-83e0-28f6d656f7b8.png"></img>

기간 : 19.08.03 ~ 19.08.04  
## 來빛 - Talk귀
 
### 팀명 : 來빛
청각장애인분들에게 소리를 빛으로 바꿔 다가감  
### 제품명 : Talk귀  
### 형태 : 토끼 모양의 무드등  
### 이용 목표층 : 청각장애인  
### 특징
평상시에는 무드등으로 사용할 수 있다.

마이크가 있어 각종 특징음(초인종 소리, 경보음, 헤어드라이기 소리, 아기 울음소리 등)이 감지되면  
스마트워치(or 제작한 팔찌)에 진동과 함께 알림이 가며  
초인종, 경보음 등 원하는 알림이 왔을 때 무드등이 함께 깜빡인다.  
```
초인종 - 녹색 3번 깜빡임  
경보음 - 붉은색 Breathing  
헤어드라이 - 푸른색 Breathing  
아기울음 - 노란색 Breathing  
```
### Raspberry Pi Environments
```
python==3.5.3  
numpy==1.17.0  
librosa==0.4.2 (some modified with 0.7.0)  
tensorflow==1.14.0  
sklearn==0.18  
bluetooth==0.22  
pyaudio==0.2.11  
```
