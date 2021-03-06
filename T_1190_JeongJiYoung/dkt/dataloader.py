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
from sklearn.model_selection import GroupShuffleSplit, train_test_split, KFold
from sklearn.preprocessing import QuantileTransformer
from torch.nn.utils.rnn import pad_sequence


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
        self.num_cols = [
            'prior_elapsed', 'mean_elapsed', 'test_time', 'grade_time', # 시간
            'answer_delta',
            'tag_delta', 'test_delta', 'assess_delta', # 정답률
            # 'big_tag_delta', 'big_test_delta', 'big_assess_delta', # 대분류&정답률
            'tag_cumAnswer'
            ]

        self.train_userID = None
        self.valid_userID = None


    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_index(self, data, train_i, valid_i):
        data_1 = [data[i] for i in train_i]
        data_2 = [data[i] for i in valid_i]

        return data_1, data_2

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        random.seed(seed) # fix to default seed 0

        if self.train_userID is not None:
            data_1 = [data[i] for i in self.train_userID]
            data_2 = [data[i] for i in self.valid_userID]
            
        elif shuffle:
            random.shuffle(data)
            size = int(len(data) * ratio)
            data_1 = data[:size]
            data_2 = data[size:]

        return data_1, data_2

    # 사용안함
    def split_specific(self, data, ratio=0.7, shuffle=True, seed=0):
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

            stratified[i] = [answer_mean]

        data_1, data_2 = train_test_split(data, train_size=ratio, stratify=stratified)

        return data_1, data_2

    def split_hardcode(self, df, ratio=0.7, shuffle=True, seed=0):
        
        df['answer_mean'] = df.groupby('userID').answerCode.transform(lambda x: x.mean())
        last_rows = df.groupby('userID').last().reset_index()
        prob = 0.
        used_ids = []
        up = last_rows[last_rows.answer_mean >= 0.610425595902904].index.tolist()
        down = last_rows[last_rows.answer_mean < 0.610425595902904].index.tolist()
        random.seed(self.args.seed)
        random.shuffle(up)
        random.shuffle(down)
        
        for _ in range(int(len(last_rows) * (1-ratio))):
            if prob >= 0.610425595902904:
                user = up.pop()
            else:
                user = down.pop()
            
            used_ids.append(user)
            prob = last_rows.iloc[used_ids].answer_mean.mean()

        self.valid_userID = used_ids
        self.train_userID = up + down
    

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
        # 대분류 추가
        df['grade'] = df['testId'].str[2]

        # 마지막 문제 이전 응답 (있을 때만)
        # last_assess = df.groupby(['userID']).assessmentItemID.last().reset_index()
        # same_with_last = df.merge(last_assess, how='inner', on=['userID', 'assessmentItemID'])
        # second_back = same_with_last[['userID', 'assessmentItemID', 'answerCode']].groupby(['userID', 'assessmentItemID']).apply(lambda x: x.iloc[-2] if len(x) > 1 else None).dropna()
        # second_back = second_back.reset_index(drop=True)
        # second_back = second_back.rename(columns={'answerCode': 'prior_answerCode'})
        # second_back = second_back.drop(columns=['assessmentItemID'])
        # df = df.merge(second_back, how='left', on=['userID'])

        # 이전 문제 소모시간 추가
        df['tmp_index'] = df.index
        tmp_df = df[['userID', 'testId', 'Timestamp', 'tmp_index']].shift(1)
        tmp_df['tmp_index'] += 1
        tmp_df = tmp_df.rename(columns={'Timestamp':'prior_timestamp'})
        df = df.merge(tmp_df, how='left', on=['userID', 'testId', 'tmp_index'])
        df['prior_elapsed'] = (df.Timestamp - df.prior_timestamp).dt.seconds

        upper_bound = df['prior_elapsed'].quantile(0.98) # outlier 설정
        median = df[df['prior_elapsed'] <= upper_bound]['prior_elapsed'].median() 
        df.loc[df['prior_elapsed'] > upper_bound, 'prior_elapsed'] = median 
        df['prior_elapsed'] = df['prior_elapsed'].fillna(median) # 빈값 채우기

        # 수치형 transform
        df['prior_elapsed'] = np.log1p(df['prior_elapsed']) #
        df['prior_elapsed'] = QuantileTransformer(output_distribution='normal').fit_transform(df.prior_elapsed.values.reshape(-1,1)).reshape(-1) 

        # 문제 평균 소모시간 추가
        assess_time = df.groupby('assessmentItemID').prior_elapsed.mean()
        assess_time.name = 'mean_elapsed'
        df = df.merge(assess_time, how='left', on=['assessmentItemID'])

        # 테스트 평균 소모시간 추가
        test_time = df.groupby('testId').prior_elapsed.mean()
        test_time.name = 'test_time'
        df = df.merge(test_time, how='left', on=['testId'])

        # 대분류별 평균 소모시간 추가
        grade_time = df.groupby('grade').prior_elapsed.mean()
        grade_time.name = 'grade_time'
        df = df.merge(grade_time, how='left', on=['grade'])

        # user&태그별 누적 카운트
        # df['tag_cumCount'] = df.groupby(['userID', 'KnowledgeTag']).cumcount()
        # df['tag_cumCount'] = np.log1p(df['tag_cumCount'])

        # user&태그별 누적 정답횟수
        df['tag_cumAnswer'] = df.groupby(['userID', 'KnowledgeTag']).answerCode.cumsum() - df['answerCode']
        df['tag_cumAnswer'] = np.log1p(df['tag_cumAnswer'])


        ### ShinChanHo
        def percentile(s):
            return np.sum(s) / len(s)
            
        # 큰 카테고리
        df['big_features'] = df['testId'].str[2].astype(int)

        # 마지막 row, answerCode -1 는 제외한 후 평균을 계산한다.
        minus_answerCode = df[df['answerCode'] == -1].index
        cal_df = df.drop(index=minus_answerCode)

        # 큰 카테고리별 정답률
        stu_groupby = cal_df.groupby('big_features').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # tag별 정답률
        stu_tag_groupby = cal_df.groupby(['KnowledgeTag']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 시험지별 정답률
        stu_test_groupby = cal_df.groupby(['testId']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 문항별 정답률
        stu_assessment_groupby = cal_df.groupby(['assessmentItemID']).agg({
            'answerCode': percentile
        }).rename(columns = {'assessmentItemID' : 'assessment_count', 'answerCode' : 'answer_rate'})

        # big&tag별 정답률
        big_tag_groupby = df.groupby(['big_features', 'KnowledgeTag']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # big&시험지별 정답률
        big_test_groupby = df.groupby(['big_features', 'testId']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # big&문항별 정답률 + 문제count
        big_assessment_groupby = df.groupby(['big_features', 'assessmentItemID']).agg({
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 정답 - 큰 카테고리별 정답률 
        '''ex)
        맞은 문제의 큰 카테고리별 정답률이 0.7 이면 1 - 0.7 = 0.3이 됨)
        틀린 문제의 큰 카테고리별 정답률이 0.7 이면 0 - 0.7 = -0.7이 됨)
        '''
        df = df.merge(stu_groupby.reset_index()[['big_features', 'answer_rate']], on=['big_features'])
        df = df.rename(columns={'answer_rate':'answer_delta'})

        # 정답 - 태그별 정답률
        df = df.merge(stu_tag_groupby.reset_index()[['answer_rate', 'KnowledgeTag']], on=['KnowledgeTag'])
        df = df.rename(columns={'answer_rate':'tag_delta'})

        # 정답 - 시험별 정답률
        df = df.merge(stu_test_groupby.reset_index()[['answer_rate', 'testId']], on=['testId'])
        df = df.rename(columns={'answer_rate':'test_delta'})

        # 정답 - 문항별 정답률
        df = df.merge(stu_assessment_groupby.reset_index()[['answer_rate', 'assessmentItemID']], on=['assessmentItemID'])
        df = df.rename(columns={'answer_rate':'assess_delta'})

        # 정답 - big & 태그별 정답률
        df = df.merge(big_tag_groupby.reset_index()[['answer_rate', 'KnowledgeTag', 'big_features']], on=['big_features', 'KnowledgeTag'])
        df = df.rename(columns={'answer_rate':'big_tag_delta'})

        # 정답 - big & 시험별 정답률
        df = df.merge(big_test_groupby.reset_index()[['answer_rate', 'testId', 'big_features']], on=['big_features', 'testId'])
        df = df.rename(columns={'answer_rate':'big_test_delta'})

        # 정답 - big&문항별 정답률
        df = df.merge(big_assessment_groupby.reset_index()[['answer_rate', 'assessmentItemID', 'big_features']], on=['big_features', 'assessmentItemID'])
        df = df.rename(columns={'answer_rate':'big_assess_delta'})

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])

        if is_train:
            # seq_len 15 이하 제외
            u_count = df.groupby('userID').assessmentItemID.count().reset_index()
            u_count = u_count.rename(columns={'assessmentItemID':'count'})
            df = df.merge(u_count, how='left', on=['userID'])
            df = df[df['count'] >= 15].reset_index()

            self.split_hardcode(df, ratio=0.8, seed=self.args.seed) # 미리 split
            extra_file_name = self.args.test_file_name
        else:
            extra_file_name = self.args.file_name
        extra_userID = None 
        # 다른 data set까지 포함(평균계산을 위해서)
        # df, extra_userID = self.add_extra_df(df, extra_file_name)

        
        # FE
        df = self.__feature_engineering(df)
        # extra_df 드롭
        if extra_userID is not None:
            df = df[~df['userID'].isin(extra_userID)]
        # Category encoding
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_cate_cols = {}
        for cate_col in self.cate_cols:
            self.args.n_cate_cols[cate_col] = len(np.load(os.path.join(self.args.asset_dir, f'{cate_col}_classes.npy')))
        self.args.n_numeric = len(self.num_cols) # 수치형은 한번에 처리

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
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

    def add_extra_df(self, df, file_name):
        extra_file_path = os.path.join(self.args.data_dir, file_name)
        extra_df = pd.read_csv(extra_file_path, parse_dates=['Timestamp'])
        extra_userID = extra_df.userID.unique()
        df = pd.concat((df, extra_df))
        return df, extra_userID


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args, is_change=False, is_reduce=False):
        self.data = data
        self.args = args
        self.is_change = is_change
        self.is_reduce = is_reduce

    def __getitem__(self, index):
        row = self.data[index] # csv의 row 형태 아님, 한 userID의 모든 데이터

        # augmentation
        if self.is_change: 
            row = self.change_seq(row)
        if self.is_reduce:
            row = self.reduce_seq(row)

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
    
    # 데이터 변경
    def change_seq(self, row):
        seq_len = len(row[0])
        aug_p = np.random.rand()
        if seq_len > self.args.max_seq_len and aug_p > 0.5:
            end_i = np.random.randint(self.args.max_seq_len, seq_len) # 최소 max_seq_len, 마지막 제외
            for i in range(len(row)):
                row[i] = row[i][:end_i]

        return row

    # 데이터 seq_len 축소
    def reduce_seq(self, row):
        seq_len = len(row[0])
        aug_p = np.random.rand()
        if seq_len > 15 and aug_p > 0.5: # 15 초과만 해당
            max_len = min(seq_len, self.args.max_seq_len)
            new_seq_len = np.random.randint(15, max_len)
            for i in range(len(row)):
                row[i] = row[i][-new_seq_len:]

        return row


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
        trainset = DKTDataset(train, args, is_change=True, is_reduce=True)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader

def get_loader(args, data, is_change=False, is_reduce=False, shuffle=False):
    pin_memory = False
    dataset = DKTDataset(data, args, is_change, is_reduce)
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=args.num_workers, shuffle=shuffle,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    return data_loader