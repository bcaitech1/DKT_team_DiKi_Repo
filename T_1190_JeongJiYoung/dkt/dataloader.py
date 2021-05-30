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

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None

        '''
        추가 가능 피쳐
        'prior_answerCode', 
        ''' 
        
        self.cate_cols = ['testId', 'assessmentItemID', 'KnowledgeTag', 'grade'] # 순서유의
        self.num_cols = ['prior_elapsed', 'mean_elapsed', 'test_time']

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)
        
        stratified = np.empty((len(data), 1))
        for i, user in enumerate(data):
            grade_mean = int(user[3].mean())
            if grade_mean > 7:
                grade_mean = 7

            answer_mean = np.round(user[-1].mean(), 1)
            if answer_mean < 0.1:
                answer_mean = 0.1

            last_answer = int(user[-1][-1]) 

            # stratified[i] = [grade_mean, answer_mean]
            stratified[i] = [last_answer]

        data_1, data_2 = train_test_split(data, train_size=ratio, stratify=stratified)

        # size = int(len(data) * ratio)
        # data_1 = data[:size]
        # data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = self.cate_cols

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                #For UNKNOWN class
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time) # too slow
        
        return df

    def __feature_engineering(self, df):
        #TODO
        # 대분류 추가
        df['grade'] = df['testId'].str[2]

        # 마지막 문제 이전 응답 (있을 때만)
        last_assess = df.groupby(['userID']).assessmentItemID.last().reset_index()
        same_with_last = df.merge(last_assess, how='inner', on=['userID', 'assessmentItemID'])
        second_back = same_with_last[['userID', 'assessmentItemID', 'answerCode']].groupby(['userID', 'assessmentItemID']).apply(lambda x: x.iloc[-2] if len(x) > 1 else None).dropna()
        second_back = second_back.reset_index(drop=True)
        second_back = second_back.rename(columns={'answerCode': 'prior_answerCode'})
        second_back = second_back.drop(columns=['assessmentItemID'])
        df = df.merge(second_back, how='left', on=['userID'])

        # 이전 문제 소모시간 추가
        df['tmp_index'] = df.index
        tmp_df = df[['userID', 'testId', 'Timestamp', 'tmp_index']].shift(1)
        tmp_df['tmp_index'] += 1
        df = df.merge(tmp_df, how='left', on=['userID', 'testId', 'tmp_index'])
        df['prior_elapsed'] = (df.Timestamp_x - df.Timestamp_y).dt.seconds

        median = df[df['prior_elapsed'] <= 260]['prior_elapsed'].median()
        df.loc[df['prior_elapsed'] > 260, 'prior_elapsed'] = median # 95%
        df['prior_elapsed'] = df['prior_elapsed'].fillna(median)

        # 문제 평균 소모시간 추가
        assess_time = df.groupby('assessmentItemID').prior_elapsed.mean()
        assess_time.name = 'mean_elapsed'
        df = df.merge(assess_time, how='left', on=['assessmentItemID'])

        # 테스트 평균 소모시간 추가
        test_time = df.groupby('testId').prior_elapsed.mean()
        test_time.name = 'test_time'
        df = df.merge(test_time, how='left', on=['testId'])

        # 수치형 로그
        df['prior_elapsed'] = np.log1p(df['prior_elapsed'])
        df['mean_elapsed'] = np.log1p(df['mean_elapsed'])
        df['test_time'] = np.log1p(df['test_time'])

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])#, nrows=100000)


        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_cate_cols = {}
        for cate_col in self.cate_cols:
            self.args.n_cate_cols[cate_col] = len(np.load(os.path.join(self.args.asset_dir, f'{cate_col}_classes.npy')))
        self.args.n_numeric = len(self.num_cols) # 수치형은 한번에 처리

        df = df.sort_values(by=['userID','Timestamp_x'], axis=0)

        group = df.groupby('userID').apply(
                lambda r: [
                    *[r[cate_col].values for cate_col in self.cate_cols],
                    *[r[num_col].values for num_col in self.num_cols],
                    r['answerCode'].values,
                ]
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index] # csv의 row 형태 아님, 한 userID의 모든 데이터

        # 각 data의 sequence length
        seq_len = len(row[0])
        cate_cols = row

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            # max seq len 보다 짧으면 남는 부분 0으로 마스크 
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1 # 뒤쪽이 시간상 끝이므로 마스크(0)가 앞부분에 생성됨.

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader