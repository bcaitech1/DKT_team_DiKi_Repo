import numpy as np
import pandas as pd
import copy
import torch
import wandb

from .metric import get_metric
from .dataloader import get_loaders, data_augmentation, DKTDataset, collate
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .trainer import get_model, train, validate, process_batch, save_checkpoint


class Trainer:
    def __init__(self):
        pass

    def train(self, args, train_data, valid_data):
        """훈련을 마친 모델을 반환한다"""

        # args update
        self.args = args

        # augmentation
        augmented_train_data = data_augmentation(train_data, args)
        if len(augmented_train_data) != len(train_data):
            print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")

        train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
        
        # only when using warmup scheduler
        args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10
            
        model = get_model(args)
        optimizer = get_optimizer(model, args)
        scheduler = get_scheduler(optimizer, args)

        best_auc = -1
        best_model = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):

            ### TRAIN
            train_auc, train_acc, loss_avg = train(train_loader, model, optimizer, scheduler, args)
            
            ### VALID
            auc, acc, preds, targets = validate(valid_loader, model, args)

            # wandb
            wandb.log({"lr": optimizer.param_groups[0]['lr'], "train_loss": loss_avg, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})

            ### model save or early stopping
            if auc > best_auc:
                best_auc = auc
                best_model = copy.deepcopy(model)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                    break

            # scheduler
            if args.scheduler == 'plateau':
                scheduler.step(best_auc)
            else:
                scheduler.step()

        return best_model

    def evaluate(self, args, model, valid_data):
        """훈련된 모델과 validation 데이터셋을 제공하면 predict 반환"""
        pin_memory = False

        valset = DKTDataset(valid_data, args)
        valid_loader = torch.utils.data.DataLoader(valset, shuffle=False,
                                                   batch_size=args.batch_size,
                                                   pin_memory=pin_memory,
                                                   collate_fn=collate)

        auc, acc, preds, _ = validate(valid_loader, model, args)
        print(f"AUC : {auc}, ACC : {acc}")

        return preds

    def test(self, args, model, test_data):
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)

        total_preds = []
        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)

            preds = model(input)

            # predictions
            preds = preds[:,-1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else: # cpu
                preds = preds.detach().numpy()
                
            total_preds.append(preds)

        total_preds = np.concatenate(total_preds)
            
        return total_preds

    def get_target(self, datas):
        targets = []
        for data in datas:
            targets.append(data[3][-1])

        return np.array(targets)


# pseudo labeling
class PseudoLabel:
    def __init__(self, trainer):
        self.trainer = trainer
        
        # 결과 저장용
        self.models =[]
        self.preds =[]
        self.valid_aucs =[]
        self.valid_accs =[]

    def visualize(self):
        aucs = self.valid_aucs
        accs = self.valid_accs

        N = len(aucs)
        auc_min = min(aucs)
        auc_max = max(aucs)
        acc_min = min(accs)
        acc_max = max(accs)

        experiment = ['base'] + [f'pseudo {i + 1}' for i in range(N - 1)]
        df = pd.DataFrame({'experiment': experiment, 'auc': aucs, 'acc': accs})

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(5 + N, 7))

        ax1.set_title('AUC of Pseudo Label Training Process', fontsize=16)

        # Time
        plt.bar(df['experiment'],
                df['auc'],
                color='red',
                width=-0.3, align='edge',
                label='AUC')
        plt.ylabel('AUC (Area Under the ROC Curve)')
        ax1.set_ylim(auc_min - 0.002, auc_max + 0.002)
        ax1.axhline(y=aucs[0], color='r', linewidth=1)
        ax1.legend(loc=2)

        # AUC
        ax2 = ax1.twinx()
        plt.bar(df['experiment'],
                df['acc'],
                color='blue',
                width=0.3, align='edge',
                label='ACC')
        plt.ylabel('ACC (Accuracy)')

        ax2.grid(False)
        ax2.set_ylim(acc_min - 0.002, acc_max + 0.002)
        ax2.axhline(y=accs[0], color='b', linewidth=1)
        ax2.legend(loc=1)

        plt.show()

    def train(self, args, train_data, valid_data):
        model = self.trainer.train(args, train_data, valid_data)

        # model 저장
        self.models.append(model)
        
        return model

    def validate(self, args, model, valid_data):
        valid_target = self.trainer.get_target(valid_data)
        valid_predict = self.trainer.evaluate(args, model, valid_data)

        # Metric
        valid_auc, valid_acc = get_metric(valid_target, valid_predict)

        # auc / acc 저장
        self.valid_aucs.append(valid_auc)
        self.valid_accs.append(valid_acc)

        print(f'Valid AUC : {valid_auc} Valid ACC : {valid_acc}')

    def test(self, args, model, test_data):
        test_predict = self.trainer.test(args, model, test_data)
        self.preds.append(test_predict)
        pseudo_labels = np.where(test_predict >= 0.5, 1, 0)
        
        return pseudo_labels

    def update_train_data(self, pseudo_labels, train_data, test_data):
        # pseudo 라벨이 담길 test 데이터 복사본
        pseudo_test_data = copy.deepcopy(test_data)

        # pseudo label 테스트 데이터 update
        for test_data, pseudo_label in zip(pseudo_test_data, pseudo_labels):
            test_data[3][-1] = pseudo_label

        # train data 업데이트
        pseudo_train_data = np.concatenate((train_data, pseudo_test_data))

        return pseudo_train_data

    def run(self, N, args, train_data, valid_data, test_data):
        """
        N은 두번째 과정을 몇번 반복할지 나타낸다.
        즉, pseudo label를 이용한 training 횟수를 가리킨다.
        """
        if N < 1:
            raise ValueError(f"N must be bigger than 1, currently {N}")
            
        # BONUS: 모델 불러오기 기능 추가
        # 별도로 관련 정답 코드는 제공되지 
        
        # pseudo label training을 위한 준비 단계
        print("Preparing for pseudo label process")
        model = self.train(args, train_data, valid_data)
        self.validate(args, model, valid_data)
        pseudo_labels = self.test(args, model, test_data)
        pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)

        # pseudo label training 원하는 횟수만큼 반복
        for i in range(N):
            print(f'Pseudo Label Training Process {i + 1}')
            model = self.train(args, pseudo_train_data, valid_data)
            self.validate(args, model, valid_data)
            pseudo_labels = self.test(args, model, test_data)
            pseudo_train_data = self.update_train_data(pseudo_labels, train_data, test_data)

            save_checkpoint({
                    'state_dict': model.state_dict(),
                    },
                    args.model_dir, 'model.pt',
                )