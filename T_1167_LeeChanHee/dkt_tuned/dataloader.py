import os
from datetime import datetime
import time
from numpy.lib.index_tricks import _diag_indices_from
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import QuantileTransformer

from .augmentation import data_augmentation

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

        self.category_cols = ['testId', 'assessmentItemID', 'KnowledgeTag', 'bigCategory'] # 범주형 컬럼
        self.numerical_cols = ['ARperCategory', 'elapsed_time', 'ARperUserID', 'AR_IdAndBig'] #수치형 컬럼

        self.train_userID = None
        self.valid_userID = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_hardcode(self, df, ratio=0.9, shuffle=True, seed=0):
        """
        정답률이 평균 이상인 유저와 평균 미만인 유저를 나누어(잘하는 / 못하는 유저),
        train/valid 셋에 골고루 넣어주기 위해, 유저들을 발라내는 작업
        """
        df['answer_mean'] = df.groupby('userID').answerCode.transform(lambda x: x.mean())
        last_rows = df.groupby('userID').last().reset_index()
        print(last_rows[1:3])
        prob = 0.
        used_ids = []
        up = last_rows[last_rows.answer_mean >= 0.6412007232826769].index.tolist()
        down = last_rows[last_rows.answer_mean < 0.6412007232826769].index.tolist()
        random.seed(self.args.seed)
        random.shuffle(up)
        random.shuffle(down)

        # up_size = int(len(up) * ratio)
        # up_data1 = up[:up_size]
        # up_data2 = up[up_size:]
        #
        # down_size = int(len(down) * ratio)
        # down_data1 = down[:down_size]
        # down_data2 = down[down_size:]
        #
        # self.train_userID = up_data1 + down_data1
        # self.valid_userID = up_data2 + down_data2

        # 정지영님 코드
        # test_data.csv의 평균 정답률 0.6104 ...
        # valid_user의 정답 평균값을 0.6104에 가깝게 맞추기 위해 prob를 반복해서 계산하여 뽑음
        for _ in range(int(len(last_rows) * (1-ratio))):
            if prob >= 0.6412007232826769:
                # 0.6411631650987256. 테스트 유저 마지막 문항 정답을 -1에서 0.5로 바꾼 후 평균낸 정답률
                #train, test 합산 정답률 0.6407346253444864
                # test셋 정답률 0.610425595902904:
                user = up.pop(0)
            else:
                user = down.pop(0)

            used_ids.append(user)
            prob = last_rows.iloc[used_ids].answer_mean.mean()

        self.valid_userID = used_ids
        self.train_userID = up + down

    def split_index(self, data, train_i, valid_i):
        data_1 = [data[i] for i in train_i]
        data_2 = [data[i] for i in valid_i]

        return data_1, data_2


    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """ split data into two parts with a given ratio """
        random.seed(seed) # fix to default seed 0

        if self.train_userID is not None: # 하드코딩 전략을 쓴 경우
            data_1 = [data[i] for i in self.train_userID]
            data_2 = [data[i] for i in self.valid_userID]

        elif shuffle: #하드코딩 전략을 쓰지 않은 경우
            random.shuffle(data)
            size = int(len(data) * ratio)
            data_1 = data[:size]
            data_2 = data[size:]

        return data_1, data_2

    def __feature_engineering(self, df):

        def percentile(s):
            return np.sum(s) / len(s)

        """FE0) 이전 문제 풀이 소모 시간"""

        """FE1) 대분류 별 정답률 계산"""
        df['bigCategory'] = df['testId'].str[2]
        #self.split_hardcode(df, ratio=0.7, seed=self.args.seed)
        #마지막 row, answerCode -1은 제외하고 평균을 계산하기
        minus_answerCode = df[df['answerCode'] == -1].index
        df = df.drop(index=minus_answerCode)

        bigCategory_groupby = df.groupby('bigCategory').agg({
            'assessmentItemID': 'count', 'answerCode': percentile
        }).rename(columns = {'assessmentItemID': 'assessItemNumber', 'answerCode' : 'answer_rate'})

        temp = pd.merge(df, bigCategory_groupby.reset_index()[['bigCategory', 'answer_rate']], on=['bigCategory'])
        temp = temp.sort_values(by=['userID', 'Timestamp']).reset_index()
        df['ARperCategory'] = temp['answer_rate']
        #0.6407346253444864

        """FE2) 대분류별 평균 소모시간 """
        # df['tmp_index'] = df.index
        # tmp_df = df[['userID', 'testId', 'Timestamp', 'tmp_index']].shift(1)
        # tmp_df['tmp_index'] += 1
        # tmp_df = tmp_df.rename(columns={'Timestamp':'prior_timestamp'})
        # df = df.merge(tmp_df, how='left', on=['userID', 'testId', 'tmp_index'])
        # df['elapsed_time'] = (df.Timestamp - df.prior_timestamp).dt.seconds

        df['tmp_index'] = df.index
        temp = df[['userID', 'testId', 'Timestamp', 'tmp_index']].shift(1)
        temp['tmp_index'] += 1
        temp = temp.rename(columns={'Timestamp': 'prior_timestamp'})
        temp.tmp_index = temp.tmp_index.fillna(0)
        temp.userID = temp.userID.fillna(0)
        temp.testId = temp.testId.fillna("A060000001")
        temp = temp.astype({'userID': 'int64', 'tmp_index': 'int64'})
        df = df.merge(temp, how='left', on=['userID', 'tmp_index', 'testId'])
        df['elapsed_time'] = (df.Timestamp - df.prior_timestamp).dt.seconds
        df = df.drop('tmp_index', axis=1)

        #이상치 기준 설정 -> 이상치 이상의 소요시간을 가진 애들은 중간값으로 처리해버림
        upper_bound = df['elapsed_time'].quantile(0.98)
        median = df[df['elapsed_time'] <= upper_bound]['elapsed_time'].median()
        df.loc[df['elapsed_time'] > upper_bound, 'elapsed_time'] = median
        df['elapsed_time'] = df['elapsed_time'].fillna(median) # nan값은 중간값으로 대치
        elapsedTimePerCategory = df.groupby('bigCategory').elapsed_time.mean()
        elapsedTimePerCategory.name = 'elapsedTimePerCategory'

        df = df.merge(elapsedTimePerCategory, how='left', on=['bigCategory'])

        """FE3) 유저별 정답률 """
        userID_groupby = df.groupby('userID').agg({
            'answerCode': percentile
        }).rename(columns={'answerCode':'ARperUserID'})

        # temp = pd.merge(df, userID_groupby.reset_index()[['userID', 'ARperUserID']], on=['userID'])
        # temp = temp.sort_values(by=['userID', 'Timestamp']).reset_index()
        # df['ARperUserID'] = temp['ARperUserID']

        df = df.merge(userID_groupby, how='left', on=['userID'])

        """FE4) 유저별 and 대분류별 정답률"""
        IdAndBig_groupby = df.groupby(['userID', 'bigCategory']).agg({
            'answerCode': percentile
            }).rename(columns={'answerCode': 'AR_IdAndBig'})

        df = df.merge(IdAndBig_groupby, how='left', on=['userID', 'bigCategory'])

        df = df.sort_values(by=['userID', 'Timestamp'])

        return df

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        #아래 3가지에 해당하는 컬럼(범주형 데이터)의 값들에 대해서 LabelEncoder를 적용해서 숫자값으로 변경
        category_cols = self.category_cols

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in category_cols:
            le = LabelEncoder()
            if is_train:
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col+'_classes.npy')
                le.classes_ = np.load(label_path)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        #     return int(timestamp)
        #
        # df['Timestamp'] = df['Timestamp'].apply(convert_time)
        return df


    def add_extra_df(self, df, extra_file_name):
        extra_file_path = os.path.join(self.args.data_dir, extra_file_name)
        extra_df = pd.read_csv(extra_file_path, parse_dates=['Timestamp'])
        # # 테스트 셋의 마지막 시퀀스 정답 코드인 -1을 평균 정답률로 바꿔주는 코드
        # extra_df.loc[extra_df.answerCode<0, 'answerCode'] = 0.6411
        extra_userID = extra_df.userID.unique()
        df = pd.concat((df, extra_df))
        return df, extra_userID

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=["Timestamp"])

        # test_data.csv를 포함하여 정답률 평균값 계산하기
        if is_train:
            extra_file_name = self.args.test_file_name
        else:
            extra_file_name = self.args.file_name

        df, extra_userID = self.add_extra_df(df, extra_file_name)

        # # augmentation 적용
        # augmented_train_data = data_augmentation(train_data, args)
        # if len(augmented_train_data) != len(train_data):
        #     print(f"Before featuring, Data Augmentation appplied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")
        #
        # augmented_valid_data = data_augmentation(valid_data, args)
        # if len(augmented_valid_data) != len(valid_data):
        #     print(f"Before featuring, Data Augmentation appplied. VALID data {len(valid_data)} -> {len(augmented_valid_data)}\n")

        df = self.__feature_engineering(df)

        if extra_userID is not None:
        # train,test의 대분류별 정답률을 구하기 위해서만 extra_userID를 사용하고, 정답률을 구한 뒤에는 기존 df에서 날려비림
            df = df[~df['userID'].isin(extra_userID)]

        self.split_hardcode(df, ratio=0.7, seed=self.args.seed)
        df = self.__preprocessing(df, is_train)

        # 추후에 피쳐 임베딩할 떄, 임베딩 레이어의 인풋 크기를 결정할 때 사용
        self.args.n_category_cols = {}
        #self.args.n_numeric_cols = {}
        for cate_col in self.category_cols:
            self.args.n_category_cols[cate_col] = len(np.load(os.path.join(self.args.asset_dir, f'{cate_col}_classes.npy')))
        # for numeric_col in self.numerical_cols:
        #     self.args.n_numeric_cols[numeric_col] = len(self.numerical_col)
        self.args.n_numeric = len(self.numerical_cols)

        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)
        group = df.groupby('userID').apply(
                lambda r: [
                    *[r[cate_col].values for cate_col in self.category_cols],
                    *[r[numerical_col].values for numerical_col in self.numerical_cols],
                    r['answerCode'].values,
                ])
        return group.values

        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'Timestamp']
        # group = df[columns].groupby('userID').apply(
        #     lambda r: (
        #         r['testId'].values,
        #         r['assessmentItemID'].values,
        #         r['KnowledgeTag'].values,
        #         r['answerCode'].values
        #     )
        # )

        # return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index] # 기존 데이터 csv의 형태가 아님. userID 단위로 구성된 데이터
        seq_len = len(row[0]) # 각 데이터의 sequence length
        categorical_cols = list(row) # row는 tuple 그래서 append가 안되기 때문에 list로 변경


        #max_sel_len을 고려해서 이보다 길면 자르고 아니면 그냥 내비둠
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(categorical_cols):
                categorical_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        #mask도 columns 목록에 포함시킴
        categorical_cols.append(mask)

        #np.array -> torch.tensor로 형변환
        for i, col in enumerate(categorical_cols):
            categorical_cols[i] = torch.tensor(col)

        return categorical_cols

    def __len__(self):
        return len(self.data)

from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    #batch 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)

def get_loaders(args, train, valid):

    pin_memory = True # gpu 사용 가능하면 True로 놓고 쓰기
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)

        #데이터 로더 사용시에 num_workers를 사용하면 파이참 디버깅에서 멈추는 버그 있음...
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=args.batch_size
                                                   ,pin_memory=pin_memory, collate_fn=collate)

    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=args.batch_size
                                                   ,pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader

