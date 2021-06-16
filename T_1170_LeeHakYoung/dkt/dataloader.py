import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None

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
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.data_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train=True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'big_features']
        for col in cate_cols:

            # #For UNKNOWN class
            # a = df[col].unique().tolist() + [np.nan]
            
            # le = LabelEncoder()
            # le.fit(a)
            # df[col] = le.transform(df[col])
            # self.__save_labels(le, col)
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                le.classes_ = np.load(label_path)
                
                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else 'unknown')

            #모든 컬럼이 범주형이라고 가정
            df[col]= df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        # df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        return df

    def __feature_engineering(self, df):
        #TODO
        def percentile(s):
            return np.sum(s) / len(s)
        
        # 큰 카테고리
        df['big_features'] = df['testId'].apply(lambda x : x[2]).astype(int)
        df['grade'] = df['testId'].str[2]

        # 큰 카테고리별 정답률
        stu_groupby = df.groupby('big_features').agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # tag별 정답률
        stu_tag_groupby = df.groupby(['big_features', 'KnowledgeTag']).agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})

        # 시험지별 정답률
        stu_test_groupby = df.groupby(['big_features', 'testId']).agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'answerCode' : 'answer_rate'})
                                                                    
        # 문항별 정답률
        stu_assessment_groupby = df.groupby(['big_features', 'assessmentItemID']).agg({
            'assessmentItemID': 'count',
            'answerCode': percentile
        }).rename(columns = {'assessmentItemID' : 'assessment_count', 'answerCode' : 'answer_rate'})

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

        # fe4 추가 feature
        # 대분류별 평균 소모시간 추가
        grade_time = df.groupby('grade').prior_elapsed.mean()
        grade_time.name = 'grade_time'
        df = df.merge(grade_time, how='left', on=['grade'])

        # user&태그별 누적 정답횟수
        df['tag_cumAnswer'] = df.groupby(['userID', 'KnowledgeTag']).answerCode.cumsum() - df['answerCode']
        df['tag_cumAnswer'] = np.log1p(df['tag_cumAnswer'])


        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        # self.args.n_questions = df['assessmentItemID'].nunique()
        # self.args.n_test = df['testId'].nunique()
        # self.args.n_tag = df['KnowledgeTag'].nunique()
        # self.args.n_big_features = 9
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_big_features = 9


        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'big_features', 'answer_delta', 'tag_delta', 'test_delta', 'assess_delta',
        'prior_elapsed', 'mean_elapsed', 'test_time', 'grade_time', 'tag_cumAnswer']
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
                    r['test_time'].values,
                    r['grade_time'].values,
                    r['tag_cumAnswer'].values
                )
            )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train = False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])

        test, question, tag, correct, big_features, answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time, grade_time, tag_cumAnswer = row
        
        # category변수와 continuout변수를 나눠줌        
        cate_cols = [test, question, tag, correct, big_features]
        cont_cols = [answer_delta, tag_delta, test_delta, assess_delta, prior_elapsed, mean_elapsed, test_time, grade_time, tag_cumAnswer]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[:seq_len] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cont_cols):
                cont_cols[i] = col[-self.args.max_seq_len:]

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cont_cols):
            cont_cols[i] = torch.tensor(col)

        return cate_cols, cont_cols

    def __len__(self):
        return len(self.data)



from torch.nn.utils.rnn import pad_sequence

def collate(batch):
    # cate변수에서 했던 처리를 cont에서도 똑같이 해줌
    cate_col_n = len(batch[0][0])
    cont_col_n = len(batch[0][1])

    cate_col_list = [[] for _ in range(cate_col_n)]
    cont_col_list = [[] for _ in range(cont_col_n)]
        
    for cate, cont in batch:
        for i, col in enumerate(cate):
            cate_col_list[i].append(col)

        for i, col in enumerate(cont):
            cont_col_list[i].append(col)

    for i, col_batch in enumerate(cate_col_list):
        cate_col_list[i] = pad_sequence(col_batch, batch_first=True)

    for i, col_batch in enumerate(cont_col_list):
        cont_col_list[i] = pad_sequence(col_batch, batch_first=True)

    # mask의 경우 max_seq_len을 기준으로 길이가 설정되어있다.
    # 만약 다른 column들의 seq_len이 max_seq_len보다 작다면
    # 이 길이에 맞추어 mask의 길이도 조절해준다
    cate_col_seq_len = cate_col_list[0].size(1)
    mask_seq_len = cate_col_list[-1].size(1)
    if cate_col_seq_len < mask_seq_len:
        cate_col_list[-1] = cate_col_list[-1][:, :cate_col_seq_len]

    cont_col_seq_len = cont_col_list[0].size(1)
    if cont_col_seq_len < mask_seq_len:
        cont_col_list[-1] = cont_col_list[-1][:, :cate_col_seq_len]

    return tuple(cate_col_list), tuple(cont_col_list) 


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None

    trainset = DKTDataset(train, args)
    valset = DKTDataset(valid, args)

    if train is not None:
        train_loader = torch.utils.data.DataLoader(trainset, shuffle=True,
                                                batch_size=args.batch_size,
                                                pin_memory=pin_memory,
                                                collate_fn=collate)

    if valid is not None:
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                batch_size=args.batch_size,
                                                pin_memory=pin_memory,
                                                collate_fn=collate)

    return train_loader, valid_loader


# Data augmentation
def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = args.stride

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if args.shuffle and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_datas = []
    for i in range(args.shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas

def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data
