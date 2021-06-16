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

**→ 이러한 문제들을 Feature / Data Augmentation / Model / Loss / Ensemble 5가지 관점으로 접근**

### Augmentation

- [Sliding window](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/fd8b99e3ae2b1ef70063a1e5eb25b981895412a5/T_1170_LeeHakYoung/dkt/dataloader.py#L339)  
  일정간격으로 Sequence length를 Shift해서 데이터를 증강합니다
- [Random Sequence Length Crop](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/db3aead0858f1042240b11434f693ecd26361945/T_1092_SeoSukMin/code/dkt/dataloader.py#L340)  
  일정확률로 Sequence Length를 축소 및 Shift시킵니다

## Model Architectures

### [bert](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1190_JeongJiYoung/dkt/model.py#L14)
BERT로 transformer 모델 첫 시도를 하였습니다.

<img src="https://user-images.githubusercontent.com/28282381/122213958-7eca8900-cee4-11eb-8249-052876f814df.png"  width=200 height=200>
### [gpt2](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1170_LeeHakYoung/dkt/model.py#L776)

오직 앞에 나오는 sequence 데이터들만 사용하여 학습을 진행하기 위한 transformer 모델입니다.
Huggingface 에서 GPT-2 모델의 구조만을 가져와 사용하였으며, GPT의 architecture는 아래의 그림과 같습니다.

<img src="https://user-images.githubusercontent.com/28282381/122216144-00231b00-cee7-11eb-9460-d214b3ae6ff5.png" width=200 height=200>
### [custom last query](https://github.com/bcaitech1/DKT_team_DiKi_Repo/blob/19f7e82bf5aab8d3b1ea5652d6227d3d0ad28f77/T_1117_ShinChanHo/code/dkt/model.py#L723)

Kagle Riiid 대회 1등 모델인 Last Query 모델에서 착안하였습니다. 
Riiid에서는 1억개의 Data가 주어졌고, 긴 Sequence로 인한 Transformer 시간복잡도를 줄이기 위해 Last Query를 사용하였습니다.
하지만 저희 DKT 대회에서는 200만개의 Data만 주어졌고, Sequence 길이에 의한 시간 복잡도는 문제 되지 않는다고 판단하여 전체 Sequence에 대해 Transformer를 적용하였습니다.
Model Architecture를 아래와 같이 도식화 하였습니다.

![image](https://user-images.githubusercontent.com/52587290/122212631-08795700-cee3-11eb-96a8-0dec6b949e3a.png)

### 최종 리더보드 점수

- ACC 0.7715 / AUC 0.8461 / 3등
