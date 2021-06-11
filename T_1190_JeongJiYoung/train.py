import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from sklearn.model_selection import KFold


def main(args):
    if args.wandb:
        wandb.login()
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    if args.wandb: 
        if args.name:
            wandb.init(project='DKT', config=vars(args), name=args.name)
        else:
            wandb.init(project='DKT', config=vars(args))

    # kfold 5
    if args.kfold5:
        kf = KFold(5, shuffle=True, random_state=args.seed)
        k = 0
        for train_i, valid_i in kf.split(train_data):
            k +=1
            train_d, valid_d = preprocess.split_index(train_data, train_i, valid_i)
            trainer.run(args, train_d, valid_d, str(k))
    else:
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data)
    
    if args.wandb: wandb.finish()

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)