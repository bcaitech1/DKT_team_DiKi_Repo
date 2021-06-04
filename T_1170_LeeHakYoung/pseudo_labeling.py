from dkt.dataloader import Preprocess, PseudoLabel
from dkt import trainer


def main(args):
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    preprocess.load_test_data(args.test_file_name)

    data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(data)
    test_data = preprocess.get_test_data()

    pseudo = PseudoLabel(trainer)

    # pseudo label 훈련!
    N = 5
    pseudo.run(N, args, train_data, valid_data, test_data)