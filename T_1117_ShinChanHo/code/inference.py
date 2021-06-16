import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


args = parse_args(mode='train')
device = "cuda" if torch.cuda.is_available() else "cpu"
args.device = device
args.n_questions = len(np.load(os.path.join(args.asset_dir,'assessmentItemID_classes.npy')))
args.n_test = len(np.load(os.path.join(args.asset_dir,'testId_classes.npy')))
args.n_tag = len(np.load(os.path.join(args.asset_dir,'KnowledgeTag_classes.npy')))
args.n_big_features = 9
model = trainer.load_model(args)
model.to(device)
args.model = model



def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    

    trainer.inference(args, test_data)
    
def gen_data(data):
    working_dir = os.getenv('WORKING_DIRECTORY')
    question_path = os.path.join(working_dir, 'server/questions.csv')
    df = pd.read_csv(question_path)

    now = datetime.now()
    df['Timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

    
    new_columns = df.columns.tolist()+['answerCode']
    new_df = pd.DataFrame([],columns=new_columns+['userID'])

    for index, row in df.iterrows():
        user_actions = pd.DataFrame(data, columns=new_columns)    
        user_actions['userID'] = index
        new_df=new_df.append(user_actions)
        row['userID'] = index
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True)

    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(int)
    new_df['userID']=new_df['userID'].astype(int)
    new_df['Timestamp']=pd.to_datetime(new_df["Timestamp"])
    print(new_df)
    print(new_df.info())
    
    return new_df


def random_gen_data(data):
    working_dir = os.getenv('WORKING_DIRECTORY')
    question_path = os.path.join(working_dir, 'server/random_questions.csv')
    df = pd.read_csv(question_path)


    now = datetime.now()
    df['Timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

    
    new_columns = df.columns.tolist()+['answerCode']
    new_df = pd.DataFrame([],columns=new_columns+['userID'])

    for index, row in df.iterrows():
        user_actions = pd.DataFrame(data, columns=new_columns)    
        user_actions['userID'] = index
        new_df=new_df.append(user_actions)
        row['userID'] = index
        new_df=new_df.append(row)
    
    new_df['answerCode'].fillna(-1, inplace=True)

    new_df['answerCode']=new_df['answerCode'].astype(int)
    new_df['KnowledgeTag']=new_df['KnowledgeTag'].astype(int)
    new_df['userID']=new_df['userID'].astype(int)
    new_df['Timestamp']=pd.to_datetime(new_df["Timestamp"])
    # print(len(new_df))
    # print(new_df.info())
    
    return new_df

def inference(data):
    
    # data = gen_data(data)
    data = random_gen_data(data)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    preprocess = Preprocess(args)
    preprocess.load_user_data(data)
    
    test_data = preprocess.get_user_data()
    
    result = trainer.user_inference(args, test_data)

    return result        
    
    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)