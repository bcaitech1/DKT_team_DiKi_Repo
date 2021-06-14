import os
from args import parse_args
from dkt_tuned.dataloader import Preprocess
from dkt_tuned import trainer
import torch
from dkt_tuned.utils import setSeeds
import wandb
from sklearn.model_selection import KFold


def main(args):

    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    args.data_dir = os.environ.get('SM_CHANNEL_TRAIN', args.data_dir)
    args.model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
#    train_data, valid_data = preprocess.split_data(train_data)

    wandb.login()
    wandb.init(project='dkt', config=vars(args))
    #네이밍 규칙 : 모델(에폭)_seq_hidden_layers_heads_Aug유무_Split_drop
    wandb_name = 'LQ(numFe4)_256_512_3_4_aug_sp0.9_drop0.3_mask'
    wandb.run.name = wandb_name


    # kfold 5
    if args.kfold5:
        kf = KFold(5, shuffle=True, random_state=args.seed)
        k = 0
        for train_i, valid_i in kf.split(train_data):
            k += 1
            train_d, valid_d = preprocess.split_index(train_data, train_i, valid_i)
            trainer.run(args, train_d, valid_d, str(k))
    else:
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data)



if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)