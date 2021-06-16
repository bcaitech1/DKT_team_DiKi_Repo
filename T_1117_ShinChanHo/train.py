import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds, increment_path
import wandb
from sklearn.model_selection import KFold

def main(args):
    wandb.login()
    
    setSeeds(args.seed)
    args.stride = args.max_seq_len

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
        
    args.model_dir = increment_path(os.path.join(args.model_dir, args.model))
    args.save_path = args.model_dir.split('/')[-1]
    os.makedirs(args.model_dir, exist_ok=True)    

    wandb.init(project='dkt', config=vars(args))
    wandb.run.name = args.name
    args = wandb.config

    if args.kfold5:
        print("kfold start!!")
        kf = KFold(5, shuffle=True, random_state=args.seed)
        k = 0
        for train_i, valid_i in kf.split(train_data):
            k += 1
            train_d, valid_d = preprocess.split_index(train_data, train_i, valid_i)
            trainer.run(args, train_d, valid_d, str(k))
    else:
        train_data, valid_data = preprocess.split_data(train_data, args.ratio)
        trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    main(args)