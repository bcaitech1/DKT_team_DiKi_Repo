import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

import random

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        # if shuffle:
        #     random.seed(seed) # fix to default seed 0
        #     random.shuffle(data)

        # size = int(len(data) * ratio)
        # data_1 = data[:size]
        # data_2 = data[size:]

        # 수정 (안나눠주고, dataset getitem에서 train은 마지막 row 떼고 학습예정)
        data_1 = data
        data_2 = data

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'big_features']

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
                
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            
        # df['time_shift'] = df['time']
        # def convert_time(s):
        #     timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
        #     return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time)   # too slow
        
        return df

    def __feature_engineering(self, df):
        # TODO
        def percentile(s):
            return np.sum(s) / len(s)


        # df['Time_shift'] = df['Time'][1:].tolist() + [-1]
        # df['Time_take'] = df['Time_shift'] - df['Time']
        # k = df['userID'][1:]
        # k = k.append(pd.Series([-1]))
        # k = k.reset_index(drop=True)
        # df.loc[df['userID'] != k, 'Time_take'] = -1000

        # 큰 카테고리
        df['big_features'] = df['testId'].apply(lambda x : x[2]).astype(int)

        # 큰 카테고리별 정답률
        file_check_path = os.path.join(self.args.asset_dir+'stu_groupby.csv')
        if os.path.isfile(file_check_path):
            stu_groupby = pd.read_csv(file_check_path).set_index('big_features')
        else:
            stu_groupby = df.groupby('big_features').agg({
                'assessmentItemID': 'count',
                'answerCode': percentile
            }).rename(columns = {'answerCode' : 'answer_rate'})
            stu_groupby.to_csv(file_check_path, index='big_features')

        # tag별 정답률
        file_check_path = os.path.join(self.args.asset_dir+'stu_tag_groupby.csv')
        if os.path.isfile(file_check_path):
            stu_tag_groupby = pd.read_csv(file_check_path).set_index(['big_features','KnowledgeTag'])
        else:
            stu_tag_groupby = df.groupby(['big_features', 'KnowledgeTag']).agg({
                'assessmentItemID': 'count',
                'answerCode': percentile
            }).rename(columns = {'answerCode' : 'answer_rate'})
            stu_tag_groupby.to_csv(file_check_path, index=['big_features','KnowledgeTag'])

        # 시험지별 정답률
        file_check_path = os.path.join(self.args.asset_dir+'stu_test_groupby.csv')
        if os.path.isfile(file_check_path):
            stu_test_groupby = pd.read_csv(file_check_path).set_index(['big_features','testId'])
        else:
            stu_test_groupby = df.groupby(['big_features', 'testId']).agg({
                'assessmentItemID': 'count',
                'answerCode': percentile
            }).rename(columns = {'answerCode' : 'answer_rate'})
            stu_test_groupby.to_csv(file_check_path, index=['big_features','testId'])
                                                                    
        # 문항별 정답률
        file_check_path = os.path.join(self.args.asset_dir+'stu_assessment_groupby.csv')
        if os.path.isfile(file_check_path):
            stu_assessment_groupby = pd.read_csv(file_check_path).set_index(['big_features','assessmentItemID'])
        else:
            stu_assessment_groupby = df.groupby(['big_features', 'assessmentItemID']).agg({
                'assessmentItemID': 'count',
                'answerCode': percentile
            }).rename(columns = {'assessmentItemID' : 'assessment_count', 'answerCode' : 'answer_rate'})
            stu_assessment_groupby.to_csv(file_check_path, index=['big_features','assessmentItemID'])

        df = df.sort_values(by=['userID','Timestamp'], axis=0)

        # 정답 - 큰 카테고리별 정답률 
        '''ex)
        맞은 문제의 큰 카테고리별 정답률이 0.7 이면 1 - 0.7 = 0.3이 됨)
        틀린 문제의 큰 카테고리별 정답률이 0.7 이면 0 - 0.7 = -0.7이 됨)
        '''
        temp = pd.merge(df, stu_groupby.reset_index()[['big_features', 'answer_rate']], on = ['big_features'])
        temp = temp.sort_values(by=['userID','Timestamp'], axis=0).reset_index()
        df['answer_delta'] = temp['answerCode'] - temp['answer_rate']

        # 정답 - 태그별 정답률
        temp = pd.merge(df, stu_tag_groupby.reset_index()[['answer_rate', 'KnowledgeTag']], on = ['KnowledgeTag'])
        temp = temp.sort_values(by=['userID','Timestamp'], axis=0).reset_index()
        df['tag_delta'] = temp['answerCode'] - temp['answer_rate']

        # 정답 - 시험별 정답률
        temp = pd.merge(df, stu_test_groupby.reset_index()[['answer_rate', 'testId']], on = ['testId'])
        temp = temp.sort_values(by=['userID','Timestamp'], axis=0).reset_index()
        df['test_delta'] = temp['answerCode'] - temp['answer_rate']

        # 정답 - 문항별 정답률
        temp = pd.merge(df, stu_assessment_groupby.reset_index()[['answer_rate', 'assessmentItemID']], on = ['assessmentItemID'])
        temp = temp.sort_values(by=['userID','Timestamp'], axis=0).reset_index()
        df['assess_delta'] = temp['answerCode'] - temp['answer_rate']

        # 이전 문제 소모시간 추가
        diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().shift(-1).fillna(pd.Timedelta(seconds=0))
        diff = diff.fillna(pd.Timedelta(seconds=0))
        diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

        df['prior_elapsed'] = diff

        upper_bound = df['prior_elapsed'].quantile(0.98)
        median = df[df['prior_elapsed'] <= upper_bound]['prior_elapsed'].median()
        df.loc[df['prior_elapsed'] > upper_bound, 'prior_elapsed'] = median
        df['prior_elapsed'] = df['prior_elapsed'].fillna(median)

        # 문제 평균 소모시간 추가
        file_check_path = os.path.join(self.args.asset_dir+'assess_time.pkl')
        if os.path.isfile(file_check_path):
            assess_time = pd.read_pickle(file_check_path)
        else:
            assess_time = df.groupby('assessmentItemID').prior_elapsed.mean()
            assess_time.to_pickle(file_check_path)
        assess_time.name = 'mean_elapsed'
        df = df.merge(assess_time, how='left', on=['assessmentItemID'])

        # 테스트 평균 소모시간 추가
        file_check_path = os.path.join(self.args.asset_dir+'test_time.pkl')
        if os.path.isfile(file_check_path):
            test_time = pd.read_pickle(file_check_path)
        else:
            test_time = df.groupby('testId').prior_elapsed.mean()
            test_time.to_pickle(file_check_path)
        test_time.name = 'test_time'
        df = df.merge(test_time, how='left', on=['testId'])

        # 수치형 로그
        df['prior_elapsed'] = np.log1p(df['prior_elapsed'])
        df['mean_elapsed'] = np.log1p(df['mean_elapsed'])
        df['test_time'] = np.log1p(df['test_time'])

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_big_features = 9

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'big_features', 'answer_delta', 
        'tag_delta', 'test_delta', 'assess_delta', 'prior_elapsed', 'mean_elapsed', 'test_time']

        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values,
                    r['big_features'].values,
                    r['answer_delta'].values,
                    r['tag_delta'].values,
                    r['test_delta'].values,
                    r['assess_delta'].values,
                    r['prior_elapsed'].values,
                    r['mean_elapsed'].values,
                    r['test_time'].values
                )
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
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct, big_features, answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time = row
        
        cate_cols = [test, question, tag, correct, big_features]
        cont_cols = [answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cont_cols):
                cont_cols[i] = col[-self.args.max_seq_len:]

        #  치팅 방지 (정답 점수 뺀거는 들어가면 안됨)
        for i, col in enumerate(cont_cols):
            cont_cols[i][-1] = 0

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)
        for i, col in enumerate(cont_cols):
            cont_cols[i] = torch.tensor(col)

        return cate_cols, cont_cols

    def __len__(self):
        return len(self.data)


class DKTDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        # 마지막 항은 TEST랑 Validation에만 있도록 만들기
        row = list(row)
       
        # AUgmentation을 넣어주자!  -> 30% 확률로 seq_len 보다 짧은 어느 랜덤 위치에서 잘라주기
        # 왜냐? train은 학습을 잘하는데 validation은 떨어짐, 과적합을 막기위해서 만들어줌
        if seq_len > 50:   # 너무 짧은 것을 자르면 좀 그러니깐 길이 100은 넘어야 자르기
            if random.random() > 0.4:   # 30% 확률로 발동
                # 앞쪽 자를 길이
                left = int((seq_len - 50) * random.random())      # 최소 80개는 되도록 자르기
                # left = 0
                
                # 뒤쪽 자를 길이
                right = int((seq_len - left - 30) * random.random())

                # 잘린 data
                seq_len = seq_len - left - right
                for i in range(len(row)):
                    row[i] = row[i][left: left + seq_len]

        test, question, tag, correct, big_features, answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time = row
        # category변수와 continuout변수를 나눠줌        
        cate_cols = [test, question, tag, correct, big_features]
        cont_cols = [answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time]


        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cont_cols):
                cont_cols[i] = col[-self.args.max_seq_len:]

        #  치팅 방지 (정답 점수 뺀거는 들어가면 안됨)
        for i, col in enumerate(cont_cols):
            cont_cols[i][-1] = 0
            
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)
        for i, col in enumerate(cont_cols):
            cont_cols[i] = torch.tensor(col)

        return cate_cols, cont_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # col_n = len(batch[0])
    cate_col_n = len(batch[0][0])
    cont_col_n = len(batch[0][1])

    # col_list = [[] for _ in range(col_n)]
    cate_col_list = [[] for _ in range(cate_col_n)]
    cont_col_list = [[] for _ in range(cont_col_n)]

    max_seq_len = len(batch[0][0][-1])    # 이거 max seq length로 되는 이유는 batch[0][0]에 마지막이 mask이기 때문임!!

    # batch의 값들을 각 column끼리 그룹화
    for cate, cont in batch:
        for i, col in enumerate(cate):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            cate_col_list[i].append(pre_padded)

        for i, col in enumerate(cont):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            cont_col_list[i].append(pre_padded)

    for i, _ in enumerate(cate_col_list):
        cate_col_list[i] =torch.stack(cate_col_list[i])

    for i, _ in enumerate(cont_col_list):
        cont_col_list[i] =torch.stack(cont_col_list[i])
    
    return tuple(cate_col_list), tuple(cont_col_list) 


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    

    if train is not None:
        trainset = DKTDatasetTrain(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader