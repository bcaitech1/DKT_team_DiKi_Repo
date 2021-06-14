import os
import gc # garbage collector
import torch
import numpy as np

from .dataloader import get_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import LSTM, LSTMATTN, LastQuery
from .augmentation import data_augmentation
import wandb

def run(args, train_data, valid_data, kfold=''):

    #캐시 메모리 비우기 및 가비지 컬렉터 가동
    torch.cuda.empty_cache()
    gc.collect()

    # augmentation 적용
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n")
    #
    # augmented_valid_data = data_augmentation(valid_data, args)
    # if len(augmented_valid_data) != len(valid_data):
    #     print(f"Data Augmentation appplied. VALID data {len(valid_data)} -> {len(augmented_valid_data)}\n")

    #train_loader, valid_loader = get_loaders(args, train_data, valid_data)
    train_loader, valid_loader = get_loaders(args, augmented_train_data, valid_data)
    #train_loader, valid_loader = get_loaders(args, augmented_train_data, augmented_valid_data)


    # Only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    args.warmup_steps = args.total_steps // 10

    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    with_acc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):
        print(f"Start Training: Epoch {epoch + 1}")
        train_auc, train_acc, train_loss = train(train_loader, model, optimizer, args)

        #Valid
        auc, acc = validate(valid_loader, model, args)


        if args.wandb:
            display_epoch = epoch + 1
            if kfold:
                display_epoch += (int(kfold)-1) * args.n_epochs
            wandb.log({"epoch": display_epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc":train_acc,
                    "valid_auc":auc, "valid_acc":acc, "lr":get_lr(optimizer)})

        #Model save or early stopping
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "train_acc": train_acc,
                   "valid_auc": auc, "valid_acc":acc})
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져온다
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint(
                {'epoch': epoch + 1, 'state_dict': model_to_save.state_dict()},
                args.model_dir,
                'model.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter +=1
            if early_stopping_counter >= args.patience:
                print(f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}")
                break

        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()



def train(train_loader, model, optimizer, args):
    model.train()
    total_preds = []
    total_targets = []
    losses = []

    for step, batch in enumerate(train_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[-4] # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        #predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)
        losses.append(loss)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses)/len(losses)
    print(f'Train auc: {auc} ACC:{acc}')
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = process_batch(batch, args)
        preds = model(input)
        targets = input[-4]

        preds = preds[:,-1]
        targets = targets[:,-1]


        if args.device == 'cuda':
            preds = preds.to('cpu').detach().numpy()
            targets = targets.to('cpu').detach().numpy()
        else:
            preds = preds.detach().numpy()
            targets = targets.detach().numpy()

        total_preds.append(preds)
        total_targets.append(targets)

    total_preds = np.concatenate(total_preds)
    total_targets = np.concatenate(total_targets)

    auc, acc = get_metric(total_targets, total_preds)
    print(f'Valid AUC: {auc} ACC: {acc}\n')

    return auc, acc


# def inference(args, test_data):
#     model = load_model(args)
#     model.eval()
#     _, test_loader = get_loaders(args, None, test_data)
#
#     total_preds = []
#
#     for step, batch in enumerate(test_loader):
#         input = process_batch(batch, args)
#         preds = model(input)
#
#         #predictions
#         preds = preds[:, -1]
#
#         if args.device == 'cuda':
#             preds = preds.to('cpu').detach().numpy()
#         else:
#             preds = preds.detach().numpy()
#
#         total_preds += list(preds)
#
#     write_path = os.path.join(args.output_dir, "output.csv")
#     if not os.path.exists(args.output_dir):
#         os.makedirs(args.output_dir)
#     with open(write_path, 'w', encoding='utf-8') as w:
#         w.write("id,prediction\n")
#         for id, p in enumerate(total_preds):
#             w.write(f'{id},{p}\n')


def inference(args, test_data):
    kfold = 5 if args.kfold5 else 1
    kfold_total = None

    for k in range(1, kfold + 1):
        if args.kfold5: args.model_name = f"model{k}.pt"
        model = load_model(args)
        model.eval()
        _, test_loader = get_loaders(args, None, test_data)
        total_preds = []

        for step, batch in enumerate(test_loader):
            input = process_batch(batch, args)
            preds = model(input)
            # predictions
            preds = preds[:, -1]

            if args.device == 'cuda':
                preds = preds.to('cpu').detach().numpy()
            else:  # cpu
                preds = preds.detach().numpy()

            total_preds += list(preds)

        if kfold_total is None:
            kfold_total = np.array(total_preds)
        else:
            kfold_total += total_preds

    # kfold 라면 평균
    if args.kfold5:
        total_preds = kfold_total / kfold

    write_path = os.path.join(args.output_dir, "output.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id, p))



def get_model(args):
    """
    Load model and move tensors to a given devices
    """
    if args.model == 'lstm': model = LSTM(args)
    if args.model == 'lstmattn': model = LSTMATTN(args)
    if args.model == 'bert': model = Bert(args)
    if args.model == 'lastquery': model = LastQuery(args)

    model.to(args.device)
    return model

def process_batch(batch, args):
    "배치를 위한 전처리 함수"
    categorical_batch = list(batch[:len(args.n_category_cols)])
    numeric_batch = list(batch[len(args.n_category_cols):-2])
    correct = batch[-2]
    mask = batch[-1]

    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor) # 해당 시퀀스의 문제를 맞췄냐 못맞췄냐 유무
    #interaction(앞에 푼 문항을 맞췃냐 못맞췃냐. 0 or 1)의 correct를 임시적으로 한칸 우측 이동 시킴
    interaction = correct + 1 # 패딩 사용을 위해. 패딩은 0 정답은 1 오답은 2
    interaction = interaction.roll(shifts=1, dims=1)
    #interaction[:, 0] = 0 # 첫번째 시퀀스에 패딩값으로 인덱스 0 부여
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:,0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    for i in range(len(categorical_batch)):
        categorical_batch[i] = ((categorical_batch[i]+1) * interaction_mask).to(torch.int64).to(args.device)
                                                            # 기존 값은 mask
    for i in range(len(numeric_batch)):
        numeric_batch[i] = (numeric_batch[i] * interaction_mask).to(torch.float).to(args.device)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1

    # 나머지 device memory로 이동
    correct = correct.to(args.device)
    mask = mask.to(args.device)
    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (*categorical_batch, *numeric_batch,
             correct, mask, interaction, gather_index)



def process_batch2(batch, args):
    test, question, tag, correct, mask = batch
    #test, question, tag, _, correct, mask = batch # LastQuery
    # change to float
    mask = mask.type(torch.FloatTensor)
    correct = correct.type(torch.FloatTensor) # 해당 시퀀스의 문제를 맞췃냐 못맞췃냐 유무 라벨(0 or 1)

    #interaction(앞에 푼 문항을 맞췃냐 못맞췃냐. 0 or 1)의 correct를 임시적으로 한칸 우측 이동 시킴
    interaction = correct + 1 # 패딩 사용을 위해. 패딩은 0 정답은 1 오답은 2
    interaction = interaction.roll(shifts=1, dims=1)
    #interaction[:, 0] = 0 # 첫번째 시퀀스에 패딩값으로 인덱스 0 부여
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:,0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)


    #  test_id, question_id, tag
    test = ((test + 1) * mask).to(torch.int64)
    question = ((question + 1) * mask).to(torch.int64)
    tag = ((tag + 1) * mask).to(torch.int64)

    # gather index
    # 마지막 sequence만 사용하기 위한 index
    gather_index = torch.tensor(np.count_nonzero(mask, axis=1))
    gather_index = gather_index.view(-1, 1) - 1


    # device memory로 이동
    test = test.to(args.device)
    question = question.to(args.device)


    tag = tag.to(args.device)
    correct = correct.to(args.device)
    mask = mask.to(args.device)

    interaction = interaction.to(args.device)
    gather_index = gather_index.to(args.device)

    return (test, question,
            tag, correct, mask,
            interaction, gather_index)

def compute_loss(preds, targets):
    loss = get_criterion(preds, targets)
    #loss = loss[:,-1] # 마지막 시퀀스에 대한 Loss 계산
    loss = torch.mean(loss)
    return loss

def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()

def save_checkpoint(state, model_dir, model_filename):
    print('saving model...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))

def load_model(args):
    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # Load model state
    model.load_state_dict(load_state['state_dict'], strict=True) # strict=True 모델이 가진 키값들이 완벽히 일치해야 함을 명

    print("Loading Model from:", model_path, "...Finished")
    return model

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']