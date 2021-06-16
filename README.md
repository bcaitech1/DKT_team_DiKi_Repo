# DKT란?

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

![f74f6ce6-dfd8-47fa-8a57-cff8c2f89c3f](https://user-images.githubusercontent.com/59329586/122187046-a2340a80-cec9-11eb-890f-895cc77b428a.png)


각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a92093c-9588-4b0e-bf06-64789b9b23e3/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5a92093c-9588-4b0e-bf06-64789b9b23e3/Untitled.png)

## 접근방법

피쳐 / Data Augmentation / 모델 / Loss / 앙상블

### 문제점

- 주어진 데이터의 feature와 user수가 너무 적음 → Feature engineering
- 버려지는 Sequence data가 너무 많다. → Data Augmentation
- 앙상블 효과를 올리기 위해서는 다양한 모델 사용이 필요하다. → 다양한 Model 학습
- Model의 output 중 꼭 마지막 문제의 output만 활용이 된다 → Loss 재정의
- AUC Metric 특성을 활용한 Ensemble → Custom Ensemble
- 모든 데이터를 훈련에 사용한다. → K-Fold CV

→ 이러한 문제들을 피쳐 / Data Augmentation / 모델 / Loss / 앙상블 5가지 관점으로 접근

### 최종 리더보드 점수

- ACC 0.7715 / AUC 0.8461 / 3등

## 팀원 소개

---

### 신찬호 T1117

- **역할**
    - Feature Engineering
    - last query
- [**github**](https://github.com/cha-no)

### 이학영 T1170

- **역할**
    - Loss function 제안
    - 새로운 loss function에 맞는 GPT-2 모델 사용
    - Bert, GPT-2 에서 최적의 hyperparameter search
- [github](https://github.com/HYLee1008), [CV](https://www.notion.so/50436491ab7a432b941c2d1f7faca6e2)

### 정지영 T1190

- **역할**
    - EDA
    - Feature Engineering
    - CV전략 찾기
    - K-Fold 적용

### 서석민 T1092

- **역할**
    - Train/Validation split 재정의
    - All Query Model
    - AUC 최대화 Ensemble기법 적용
- [**Notion**](https://www.notion.so/T_1092-4e5450cc96e04326a8377fbffc788376), [**Github**](https://github.com/min1321)
