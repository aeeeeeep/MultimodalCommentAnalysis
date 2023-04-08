WANDB=True
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import BertDataset
from models import CustomModel
from utils import to_device, Checkpoint, Step, Smoother, Logger
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, accuracy_score
import warnings
warnings.filterwarnings("ignore")

def compute_batch(model, source, target):
    source = source.to('cuda:0')
    target = to_device(target, 'cuda:0')
    pred = model(source)
    loss = losses(target, pred)
    return loss, pred

def evaluate(model, loader):
    metrics = Smoother(100)
    accuracy_list = []
    train_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    mse_list = []
    mae_list = []

    for (source, target) in tqdm(loader):
        source = source.to('cuda:0')
        pred = model(source)
        pred = pred.cpu().detach()

        accuracy = accuracy_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
        precision = precision_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
        recall = recall_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
        f1 = f1_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
        try:
            roc_auc = roc_auc_score(target.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
        except ValueError:
            pass
        mse = mean_squared_error(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
        mae = mean_absolute_error(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        roc_auc_list.append(roc_auc)
        mse_list.append(mse)
        mae_list.append(mae)

    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_roc_auc = np.mean(roc_auc_list)
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)

    if WANDB:
        wandb.log({
            "val_accuracy":avg_accuracy,
            "val_precision":avg_precision,
            "val_recall":avg_recall,
            "val_f1":avg_f1,
            "val_roc_auc":avg_roc_auc,
            "val_mse":avg_mse,
            "val_mae":avg_mae,
            })

    metrics.update(val_acc=avg_accuracy)
    return metrics

def get_model():
    return CustomModel()

losses = nn.BCEWithLogitsLoss()
def train():
    if WANDB:
        wandb.init(
            project="MultimodalCommentAnalysis",
            name="bert",
        )

    train_data = BertDataset(conf['train_file'], conf['input_l'])
    valid_data = BertDataset(conf['valid_file'], conf['input_l'])

    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model=model, step=step)
    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])

    start_epoch = 0

    logger = Logger(conf['model_dir'] + '/log%d.txt' % version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['model_dir'])

    Path(conf['model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['n_epoch']):
        accuracy_list = []
        train_acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        mse_list = []
        mae_list = []

        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for (source, target) in tqdm(train_loader):
            step.forward(conf['batch'])

            loss, pred = compute_batch(model, source, target)
            loss = loss.mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            logger.log(step.value, loss.item())
            accuracy = accuracy_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            recall = recall_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(target.argmax(dim=1).cpu().numpy(), F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
            except ValueError:
                pass
            mse = mean_squared_error(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            mae = mean_absolute_error(target.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
            
            if WANDB:
                wandb.log({'step': step.value,'train_loss': loss.item()})
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            roc_auc_list.append(roc_auc)
            mse_list.append(mse)
            mae_list.append(mae)

        avg_accuracy = np.mean(accuracy_list)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        avg_roc_auc = np.mean(roc_auc_list)
        avg_mse = np.mean(mse_list)
        avg_mae = np.mean(mae_list)

        if WANDB:
            wandb.log({
                "train_accuracy":avg_accuracy,
                "train_precision":avg_precision,
                "train_recall":avg_recall,
                "train_f1":avg_f1,
                "train_roc_auc":avg_roc_auc,
                "train_mse":avg_mse,
                "train_mae":avg_mae,
                })

        if epoch%1 == 0:
            checkpoint.save(conf['model_dir'] + '/model_%d.pt' % epoch)
            model.eval()
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            writer.add_scalars('valid acc', metrics.value(), step.value)
            checkpoint.update(conf['model_dir'] + '/model.pt', metrics=metrics.value())
            model.train()

        if WANDB:
            wandb.log({'epoch': epoch})
    logger.close()
    writer.close()


version = 1
conf = Config(version)

train()
