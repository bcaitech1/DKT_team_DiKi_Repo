import wandb
import os
from dkt.dataloader import Preprocess
from dkt.pseudolabel import PseudoLabel, Trainer
from dkt.utils import setSeeds
from args import parse_args



def main(args):
    wandb.login()

    setSeeds(args.seed)
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    preprocess.load_test_data(args.test_file_name)

    data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data)
    test_data = preprocess.get_test_data()

    trainer = Trainer()

    pseudo = PseudoLabel(trainer)

    wandb.init(project='dkt_newstart', config=vars(args))
    # pseudo label 훈련!
    N = 5
    pseudo.run(N, args, train_data, valid_data, test_data)


if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)