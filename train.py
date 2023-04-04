WANDB = True
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import BertDataset
from models import CustomModel
from utils import to_device, Checkpoint, Step, Smoother, Logger, EMA, FGM


def compute_batch(model, source, targets):
    # source = to_device(source, 'cuda:0')
    source = source.to('cuda:0')
    targets = to_device(targets, 'cuda:0')
    pred = model(source)
    loss = losses(targets, pred)
    return loss

def evaluate(model, loader):
    metrics = Smoother(100)
    val_acc = 0
    tot_count = 0
    for (source, targets) in tqdm(loader):
        # source = to_device(source, 'cuda:0')
        source = source.to('cuda:0')
        pred = model(source)
        pred = pred.cpu().detach()
        val_acc += pred.argmax(dim=1).eq(targets.argmax(dim=1)).sum().item()
        tot_count += targets.shape[0]
    metrics.update(val_acc=val_acc / tot_count)
    print(metrics.value())
    return metrics

def get_model():
    return CustomModel()

losses = nn.CrossEntropyLoss()
def train():
    if WANDB:
        wandb.init(
            project="MultimodalCommentAnalysis",
            name="bart",
        )

    train_data = BertDataset(conf['train_file'], conf['input_l'])
    valid_data = BertDataset(conf['valid_file'], conf['input_l'])

    train_loader = DataLoader(train_data, batch_size=conf['batch'], shuffle=True, num_workers=12, drop_last=False)
    valid_loader = DataLoader(valid_data, batch_size=conf['valid_batch'], shuffle=True, num_workers=12, drop_last=False)

    model = get_model()
    step = Step()
    checkpoint = Checkpoint(model=model, step=step)
    model = torch.nn.DataParallel(model)
    ema = EMA(model, 0.999, device="cuda:0")
    ema.register()
    fgm = FGM(model)
    model.to('cuda:0')

    optimizer = torch.optim.AdamW(model.parameters(), lr=conf['lr'])

    start_epoch = 0

    logger = Logger(conf['model_dir'] + '/log%d.txt' % version, 'a')
    logger.log(conf)
    writer = SummaryWriter(conf['model_dir'])

    Path(conf['model_dir']).mkdir(exist_ok=True, parents=True)
    for epoch in range(start_epoch, conf['n_epoch']):
        print('epoch', epoch)
        logger.log('new epoch', epoch)
        for (source, targets) in tqdm(train_loader):
            step.forward(conf['batch'])

            loss = compute_batch(model, source, targets)
            loss = loss.mean()
            loss.backward()

            fgm.attack()
            adv_loss = compute_batch(model, source, targets)
            adv_loss = adv_loss.mean()
            adv_loss.backward()
            fgm.restore()

            optimizer.step()
            ema.update()
            optimizer.zero_grad()

            if step.value % 100 == 0:
                logger.log(step.value, loss.item())
                if WANDB:
                    wandb.log({'step': step.value})
                    wandb.log({'train_loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        ema.apply_shadow()
        if epoch % 1 == 0:
            checkpoint.save(conf['model_dir'] + '/model_%d.pt' % epoch)
            model.eval()
            metrics = evaluate(model, valid_loader)
            logger.log('valid', step.value, metrics.value())
            if WANDB:
                wandb.log({'valid_acc': metrics.value()})
            writer.add_scalars('valid acc', metrics.value(), step.value)
            checkpoint.update(conf['model_dir'] + '/model.pt', metrics=metrics.value())
            model.train()

        if WANDB:
            wandb.log({'epoch': epoch})
        ema.restore()
    logger.close()
    writer.close()


version = 1
conf = Config(version)

train()
