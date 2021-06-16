import os
import argparse

def str2bool(v : str) -> bool:
    """convert string argument to boolean
    Args:
        v (str)
    Raises:
        argparse.ArgumentTypeError: [Boolean value expected]
    Returns:
        bool
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int, help='seed')    
    parser.add_argument('--port', default=6007, type=int, help='seed')

    parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')

    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='asset/', type=str, help='data directory')
    
    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    
    parser.add_argument('--model_dir', default='/opt/ml/models/', type=str, help='model directory')
    parser.add_argument('--model_name', default='model.pt', type=str, help='model file name')

    parser.add_argument('--output_dir', default='output/', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')
    parser.add_argument('--name', default='fe10_maxseq200_hiddendim128_gpt2_new')
    parser.add_argument('--ratio', default=0.7, type=float, help='train rate')

    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')
    parser.add_argument('--max_seq_len', default=200, type=int, help='max sequence length')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')

    parser.add_argument('--window', default=True, type=str2bool)
    parser.add_argument('--shuffle', default=False, type=str2bool)

    # 모델
    parser.add_argument('--n_cate', default=5, type=int)
    parser.add_argument('--n_cont', default=9, type=int)
    parser.add_argument('--shuffle_n', default=2, type=int)
    parser.add_argument('--hidden_dim', default=128, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=1, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=16, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.0, type=float, help='drop out rate')
    
    # T Fixup
    parser.add_argument('--Tfixup', default=True, type=str2bool)
    parser.add_argument('--layer_norm', default=True, type=str2bool)

    # 훈련
    parser.add_argument('--n_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=15, type=int, help='for early stopping')
        

    ### 중요 ###
    parser.add_argument('--model', default='gpt2', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='plateau', type=str, help='scheduler type')
    parser.add_argument('--kfold5', default=False, type=str2bool)


    args = parser.parse_args()

    return args