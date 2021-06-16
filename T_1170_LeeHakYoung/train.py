import os
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
from sklearn.model_selection import KFold


# report에서 auc및 실행 시간 정보 얻기
def time_auc(report, n_epoch=10):
    total_time = 0
    for epoch in range(1, n_epoch + 1):
        result = report[str(epoch)]
        total_time += result['train_time']
        total_time += result['valid_time']

    return total_time, report['best_auc'], report['best_acc']


def main(args):
    wandb.login()
    
    setSeeds(args.seed)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)

    train_data = preprocess.get_train_data()
    
    # train_data, valid_data = preprocess.split_data(train_data)

    # print(f"훈련(train) 데이터 준비 완료 : {len(train_data)} 개")
    # print(f"검증(valid) 데이터 준비 완료 : {len(valid_data)} 개")

    # columns = ['testId', 'assessmentItemID', 'KnowledgeTag', 'answerCode']
    # user_data = train_data[0]

    # # 각 데이터에 어떤 값이 들어있는지 체크해보자
    # for column, data in zip(columns, user_data):
    #     print(f'{column:18} : {data[:10]}')

    wandb.init(project='dkt_newstart', config=vars(args))

        # kfold 5
    if args.kfold:
        kf = KFold(5, shuffle=True, random_state=args.seed)
        k = 0
        for train_i, valid_i in kf.split(train_data):
            k +=1
            train_d, valid_d = preprocess.split_index(train_data, train_i, valid_i)
            _ = trainer.run(args, train_d, valid_d, str(k))
    else:
        train_data, valid_data = preprocess.split_data(train_data)
        _ = trainer.run(args, train_data, valid_data)
    # total_time, auc, _ = time_auc(report)

    # print(f"Cost Time : {total_time} sec, best AUC : {auc}")

    # print(f"AUC : {report['best_auc']} at epoch {report['best_auc_epoch']}")
    # print(f"ACC : {report['best_acc']} at epoch {report['best_acc_epoch']}")

    # print(f"Train time at [ epoch 5 ] : {report['5']['train_time']:.2f} 초")
    # print(f"Validate time at [ epoch 5 ] : {report['5']['valid_time']:.2f} 초")

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)