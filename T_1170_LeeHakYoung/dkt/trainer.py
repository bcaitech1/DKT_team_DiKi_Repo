import os
import torch
import numpy as np
import gc
from collections import OrderedDict
import time


from .dataloader import get_loaders, data_augmentation
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, Bert, Saint, LastQuery, FixupEncoder, TfixupBert

import wandb

def run(args, train_data, valid_data, gradient=False):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()

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

    # 🌟 분석에 사용할 값 저장 🌟
    # report = OrderedDict()

    # gradient step 분석에 사용할 변수
    if gradient:
        args.n_iteration = 0
        args.gradient = OrderedDict()

        # 모델의 gradient값을 가리키는 모델 명 저장
        args.gradient['name'] = [name for name, _ in model.named_parameters()]

    best_auc = -1
    best_auc_epoch = -1
    best_acc = -1
    best_acc_epoch = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        # epoch_report = {}

        print(f"Start Training: Epoch {epoch + 1}")
        
        ### TRAIN
        train_start_time = time.time()
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, scheduler, args, gradient)
        train_time = time.time() - train_start_time

        # epoch_report['train_auc'] = train_auc
        # epoch_report['train_acc'] = train_acc
        # epoch_report['train_time'] = train_time
        
        ### VALID
        valid_start_time = time.time()
        auc, acc,_ , _ = validate(valid_loader, model, args)
        valid_time = time.time() - valid_start_time

        # epoch_report['valid_auc'] = auc
        # epoch_report['valid_acc'] = acc
        # epoch_report['valid_time'] = valid_time

        # save lr
        # epoch_report['lr'] = optimizer.param_groups[0]['lr']

        # 🌟 save it to report 🌟
        # report[f'{epoch + 1}'] = epoch_report

        ### TODO: model save or early stopping
        wandb.log({"lr": optimizer.param_groups[0]['lr'], "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                  "valid_auc":auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.model_dir, 'model.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)

    # save best records
    # report['best_auc'] = best_auc
    # report['best_auc_epoch'] = best_auc_epoch
    # report['best_acc'] = best_acc
    # report['best_acc_epoch'] = best_acc_epoch

    # save gradient informations
    if gradient:
        # report['gradient'] = args.gradient
        del args.gradient
        del args['gradient']

    return None #report


def train(train_loader, model, optimizer, scheduler, args, gradient=False):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[0][3] # correct
        index = input[0][-1] # gather index
        
        loss = compute_loss(preds, targets, index)
        loss.backward()

        # save gradient distribution
        if gradient:
            args.n_iteration += 1
            args.gradient[f'iteration_{args.n_iteration}'] = get_gradient(model)

        # grad clip
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        optimizer.zero_grad()

        # warmup scheduler
        if args.scheduler == 'linear_warmup':
            scheduler.step()

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")
        
        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()
        
        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)
      

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'TRAIN AUC : {auc} ACC : {acc}')

    return auc, acc, loss_avg
    

def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)

        preds = model(input)
        targets = input[0][3] # correct
        index = input[0][-1] # gather index


        # predictions
        preds = preds.gather(1, index).view(-1)
        targets = targets.gather(1, index).view(-1)
    
        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    
    print(f'VALID AUC : {auc} ACC : {acc}\n')

    return auc, acc, total_preds, total_targets



def inference(args, test_data):
    
    model = load_model(args)
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)
    
    
    total_preds = []
    
    for step, batch in enumerate(test_loader):
        input = process_batch(batch, args)
        index = input[0][-1]

        preds = model(input)
        

        # predictions
        preds = preds.gather(1, index).view(-1)
        

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
        else: # cpu
            preds = preds.detach().numpy()
            
        total_preds+=list(preds)

    write_path = os.path.join(args.output_dir, "output.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))




def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'last_query': model = LastQuery(args)
    if args.model == 'saint': model = Saint(args)
    if args.model == 'tfixup': model = FixupEncoder(args)
    if args.model == 'tfixupbert': model = TfixupBert(args)

    model.to(args.device)

    return model


# 배치 전처리
def process_batch(batch, args):

    (test, question, tag, correct, big_features, mask), cont_features = batch    
    
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor)
    big_features = big_features.type(torch.FloatTensor)

    temp = []

    interaction = correct + 1 # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction[:, 0] = 0 # set padding index to the first sequence
    interaction = (interaction * mask).to(torch.int64)
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)
    big_features = (big_features * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # interaction과 동일하게 rolling을 해서 이전 정보를 사용할 수 있도록 함
    for cont_feature in cont_features:
        cont_feature = cont_feature.type(torch.FloatTensor)
        cont_feature = cont_feature.roll(shifts=1, dims=1)
        cont_feature[:, 0] = 0
        cont_feature = (cont_feature * mask).unsqueeze(-1)
        temp.append(cont_feature)
    
    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)
    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)
    big_features = big_features.to(args.device)
    gather_index = gather_index.to(args.device)

    # 연속형 변수들을 concat해줌
    cont_features = torch.cat(temp, dim=-1).to(args.device)

    return (test, question,
            tag, correct, mask,
            interaction, big_features, gather_index), cont_features


# loss계산하고 parameter update!
def compute_loss(preds, targets, index):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)
        index    : (batch_size, max_seq_len)

        만약 전체 sequence 길이가 max_seq_len보다 작다면 해당 길이로 진행
    """
    loss = get_criterion(preds, targets)
    # loss = torch.gather(loss, 1, index)
    loss = torch.mean(loss)

    return loss


def save_checkpoint(state, model_dir, model_filename):
    print('saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))



def load_model(args):
    
    
    model_path = os.path.join(args.model_dir,'model.pt')
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # 1. load model state
    model.load_state_dict(load_state['state_dict'], strict=True)
   
    
    print("Loading Model from:", model_path, "...Finished.")
    return model


def get_gradient(model):
    gradient = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad != None:
            gradient.append(grad.cpu().numpy().astype(np.float16))
            # gradient.append(grad.clone().detach())
        else:
            gradient.append(None)

    return gradient