import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import random
import wandb
def main(args):
    wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    # user split for validation
    args.train_split = True
    args.validation = False
    args.all_user = set(range(7442))
    args.val_user = random.sample(args.all_user, int(7442/5))

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    args.file_name = args.file_name_val
    args.validation = True

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    valid_data = preprocess.get_train_data()    
    
    # train_data, valid_data = preprocess.split_data(train_data)

    del args.all_user
    del args.val_user
    
    wandb.init(project='dkt', name=args.name, config=vars(args))

    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)