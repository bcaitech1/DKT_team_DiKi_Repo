# Team DiKi Deep Knowledge Detection(DKT)

# 목차 

- [프로젝트 소개](#프로젝트-소개)
- [문제점](#문제점)
- [Model](#model-architectures)
- [최종 리더보드 점수](#최종-리더보드-점수)


## 프로젝트 소개

### DKT란?

DKT는 Deep Knowledge Tracing의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

![f74f6ce6-dfd8-47fa-8a57-cff8c2f89c3f](https://user-images.githubusercontent.com/59329586/122187046-a2340a80-cec9-11eb-890f-895cc77b428a.png)


각 학생이 푼 문제 리스트와 정답 여부가 담긴 데이터를 받아 최종 문제를 맞출지 틀릴지 예측

![Untitled](https://user-images.githubusercontent.com/59329586/122187219-ce4f8b80-cec9-11eb-9a09-b2ad63b61155.png)

### DATA

train/test 합쳐서 총 7,442명의 사용자가 존재합니다. 
사용자가 푼 마지막 문항의 정답을 맞출 것인지 예측하는 것이 최종 목표입니다.

![dataframe](https://user-images.githubusercontent.com/59329586/122188883-61d58c00-cecb-11eb-9d12-aefb75c3b7a2.png)

- userID 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 userID를 기준으로 90/10의 비율로 나누어졌습니다.  
- assessmentItemID 문항의 고유번호입니다.  
- testId 시험지의 고유번호입니다.  
- answerCode 사용자가 해당 문항을 맞췄는지 여부입니다. 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.
- Timestamp 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.
- KnowledgeTag 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다.
- Test data의 가장 마지막에 푼 문항의 answerCode는 모두 -1로 표시되어 있습니다. 대회 목표는 이 -1로 처리되어 있는 interaction의 정답 여부를 맞추는 것입니다.


### 평가방법
DKT는 주어진 마지막 문제를 맞았는지 틀렸는지로 분류하는 이진 분류 문제입니다! 
그래서 평가를 위해 AUROC(Area Under the ROC curve)를 사용합니다

![047c8d8e-5d30-4d5a-8afd-d72a254c6318](https://user-images.githubusercontent.com/59329586/122188993-7dd92d80-cecb-11eb-9e3d-53bec5db329d.png)

## 문제점

- 주어진 데이터의 feature와 user수가 너무 적음 → [Feature engineering](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/main/T_1170_LeeHakYoung/dkt/dataloader.py#L83)
- 버려지는 Sequence data가 너무 많다. → [Data Augmentation](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/main/T_1170_LeeHakYoung/dkt/dataloader.py#L339)
- 앙상블 효과를 올리기 위해서는 다양한 모델 사용이 필요하다. → [다양한 Model 학습](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/main/T_1170_LeeHakYoung/dkt/model.py)
- model의 output 중 꼭 마지막 문제의 output만 활용이 된다 → [Loss 재정의](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/main/T_1170_LeeHakYoung/dkt/trainer.py#L301)
- AUC Metric 특성을 활용한 Ensemble → Custom Ensemble
- 모든 데이터를 훈련에 사용한다. → [K-Fold CV](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/04a0235fe86e9eb04aa3372d71c34a41229bfc09/T_1190_JeongJiYoung/train.py#L30)

→ 이러한 문제들을 Feature / Data Augmentation / Model / Loss / Ensemble 5가지 관점으로 접근

### Augmentation

- Sliding window
- Random Sequence Length Crop

## Model Architectures

- [bert](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1190_JeongJiYoung/dkt/model.py#L14)
- [gpt2](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1170_LeeHakYoung/dkt/model.py#L776)
- [custom last query](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1117_ShinChanHo/code/dkt/model.py#L723)


### 최종 리더보드 점수

- ACC 0.7715 / AUC 0.8461 / 3등
