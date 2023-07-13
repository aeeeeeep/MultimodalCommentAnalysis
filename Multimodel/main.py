WANDB = True
import logging
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, \
    mean_absolute_error, accuracy_score
from tqdm import tqdm

from config import parse_args
from data_helper import create_dataloaders
from model import MultiModal
from util import setup_device, setup_seed, setup_logging, build_optimizer

warnings.filterwarnings("ignore")


def validate(model, val_dataloader):
    model.eval()
    accuracy_list = []
    train_loss_list = []
    train_acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    roc_auc_list = []
    mse_list = []
    mae_list = []
    best_score = 0.

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            loss, pred, labels = model(batch)
            loss = loss.mean()
            accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(),
                                        average='macro')
            recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
            try:
                roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(),
                                        F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
            except ValueError:
                pass
            mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
            mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
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
            "val_accuracy": avg_accuracy,
            "val_precision": avg_precision,
            "val_recall": avg_recall,
            "val_f1": avg_f1,
            "val_roc_auc": avg_roc_auc,
            "val_mse": avg_mse,
            "val_mae": avg_mae,
        })

    if avg_f1 > best_score:
        best_score = avg_f1
        state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
        torch.save({'epoch': epoch, 'model_state_dict': state_dict, 'f1': avg_f1},
                   f'{args.savedmodel_path}/model_epoch_{epoch}_f1_{avg_f1}.bin')

    model.train()
    return


def train_and_validate(args):
    if WANDB:
        wandb.init(project="MultimodalCommentAnalysis", name="Multimodel")

    # 1. load data
    train_dataloader, val_dataloader = create_dataloaders(args)

    # 2. build model and optimizers
    model = MultiModal(args)
    optimizer, scheduler = build_optimizer(args, model)
    if args.device == 'cuda':
        model = torch.nn.parallel.DataParallel(model.to(args.device))

    # 3. training
    step = 0
    best_score = args.best_score
    start_time = time.time()
    num_total_steps = len(train_dataloader) * args.max_epochs
    for epoch in range(args.max_epochs):
        accuracy_list = []
        train_acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        roc_auc_list = []
        mse_list = []
        mae_list = []
        for batch in tqdm(train_dataloader):
            model.train()
            loss, pred, labels = model(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            step += 1
            if step % args.print_steps == 0:
                # time_per_step = (time.time() - start_time) / max(1, step)
                # remaining_time = time_per_step * (num_total_steps - step)
                # remaining_time = time.strftime('%H:%M:%S', time.gmtime(remaining_time))
                # logging.info(f"Epoch {epoch} step {step} eta {remaining_time}: loss {loss:.3f}, accuracy {accuracy:.3f}")
                if WANDB:
                    wandb.log({
                        "step": step,
                        "train_loss": loss,
                    })

                accuracy = accuracy_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy())
                precision = precision_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(),
                                            average='macro')
                recall = recall_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(),
                                      average='macro')
                f1 = f1_score(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().numpy(), average='macro')
                try:
                    roc_auc = roc_auc_score(labels.argmax(dim=1).cpu().numpy(),
                                            F.softmax(pred, dim=1).detach().cpu().numpy()[:, 1])
                except ValueError:
                    pass
                mse = mean_squared_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())
                mae = mean_absolute_error(labels.argmax(dim=1).cpu().numpy(), pred.argmax(dim=1).cpu().detach().numpy())

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
                "train_accuracy": avg_accuracy,
                "train_precision": avg_precision,
                "train_recall": avg_recall,
                "train_f1": avg_f1,
                "train_roc_auc": avg_roc_auc,
                "train_mse": avg_mse,
                "train_mae": avg_mae,
            })

        # 4. validation
        validate(model, val_dataloader)
        # results = {k: round(v, 4) for k, v in results.items()}
        # logging.info(f"Epoch {epoch} step {step}: loss {loss:.3f}, {results}")


def main():
    args = parse_args()
    setup_logging()
    setup_device(args)
    setup_seed(args)

    os.makedirs(args.savedmodel_path, exist_ok=True)
    logging.info("Training/evaluation parameters: %s", args)

    train_and_validate(args)


if __name__ == '__main__':
    main()
