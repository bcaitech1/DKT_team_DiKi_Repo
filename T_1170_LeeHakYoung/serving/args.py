import torch
import easydict


def parse_args(mode='train'):
    config = {}

    # 설정
    config['seed'] = 42
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['data_dir'] = '/opt/ml/input/data/train_dataset'
    config['model_dir'] = 'models/'
    config['file_name'] = 'train_data.csv'
    config['log_steps'] = 50
    config['patience'] = 15

    # 데이터
    config['max_seq_len'] = 200

    # 데이터 증강 (Data Augmentation)
    config['window'] = True
    config['stride'] = config['max_seq_len']
    config['shuffle'] = False
    config['shuffle_n'] = 2

    # 모델
    config['hidden_dim'] = 128
    config['n_layers'] = 1
    config['dropout'] = 0.0
    config['n_heads'] = 16

    # T Fixup
    config['Tfixup'] = True
    config['layer_norm'] = True

    # 훈련
    config['n_epochs'] = 100
    config['batch_size'] = 64
    config['lr'] = 0.0001
    config['clip_grad'] = 10

    # inference
    config['test_file_name'] = 'test_data.csv'
    config['asset_dir'] = 'asset/'
    config['output_dir'] = 'output/'

    ### 중요 ###
    config['model'] = 'gpt2'
    config['optimizer'] = 'adamW'
    config['scheduler'] = 'plateau'

    args = easydict.EasyDict(config)
    return args