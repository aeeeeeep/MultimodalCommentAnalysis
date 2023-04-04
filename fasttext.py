WANDB = True
import wandb
import random
from collections import Counter, OrderedDict
import os
import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import ngrams_iterator
from torchtext.transforms import VocabTransform
from torchtext.vocab import vocab
from tqdm import tqdm

from utils import *


class Args:
    def __init__(self) -> None:
        self.batch_size = 8
        self.lr = 0.5
        self.epochs = 12
        self.num_workers = 12

        self.embed_size = 256
        self.hidden_size = 32
        self.output_size = 2
        self.seed = 42

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


args = Args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class Net(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, args.embed_size)
        self.linear = nn.Linear(args.embed_size, args.output_size)

    def forward(self, text_token):
        embedded = self.embedding(text_token)
        pooled = nn.functional.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        out_put = self.linear(pooled)
        return out_put

    def get_embedding(self, token_list: list):
        return self.embedding(torch.Tensor(token_list).long())


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Dataset(Dataset):
    def __init__(self, path, flag='train', max_length=256) -> None:
        self.df = pd.read_csv(path)
        self.text_list = self.df['text'].values
        self.text_label = self.df['label'].values
        # self.token = process_notes(self.text_list)
        self.max_length = max_length
        self.text_vocab, self.vocab_transform = self.reform_vocab(self.text_list)
        self.fast_data = self.generate_fast_text_data()
        self.len = len(self.text_list)

    def __getitem__(self, index):
        data_row = self.fast_data[index]
        data_row = pad_or_cut(data_row, self.max_length)
        data_label = torch.zeros(2, dtype=torch.float32)
        data_label[int(self.text_label[index])] = 1
        return data_row, data_label

    def __len__(self) -> int:
        return self.len

    def get_labels(self):
        return self.text_label

    def reform_vocab(self, text_list):
        total_word_list = []
        for _ in text_list:
            total_word_list += list(ngrams_iterator(_.split(" "), 2))
        counter = Counter(total_word_list)
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)
        special_token = ["<UNK>", "<SEP>"]
        text_vocab = vocab(ordered_dict, specials=special_token)
        text_vocab.set_default_index(0)
        vocab_transform = VocabTransform(text_vocab)
        return text_vocab, vocab_transform

    def generate_fast_text_data(self):
        fast_data = []
        for sentence in self.text_list:
            all_sentence_words = list(ngrams_iterator(sentence.split(' '), 2))
            sentence_id_list = np.array(self.vocab_transform(all_sentence_words))
            fast_data.append(sentence_id_list)
        return fast_data

    def get_vocab_transform(self):
        return self.vocab_transform

    def get_vocab_size(self):
        return len(self.text_vocab)


def train():
    if WANDB:
        wandb.init(
                project="MultimodalCommentAnalysis",
                name="fasttext",
                )
    train_dataset = Dataset("./data/train.csv", flag='train')
    train_dataloader = DataLoaderX(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=False)
    val_dataset = Dataset("./data/val.csv", flag='val')
    val_dataloader = DataLoaderX(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, drop_last=True)

    model = Net(train_dataset.get_vocab_size()).to(args.device)
    corss_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1))
    best_acc = 0.

    for epoch in range(args.epochs):
        print('Epoch: ', epoch)
        train_loss = 0.
        train_acc = 0
        tot_count = 0
        model.train()
        for idx, (data, target) in enumerate(tqdm(train_dataloader)):
            data, target = data.to(args.device), target.to(args.device)
            pred = model(data)
            loss = corss_loss(pred, target)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_acc += pred.argmax(dim=1).eq(target.argmax(dim=1)).sum().item()
            tot_count += target.shape[0]
            if WANDB:
                wandb.log({"train_loss": loss.item()})

        print('train Loss:{:.3f} Acc: {:.2f}% {}/{} lr:{:.2e}'.format(
            train_loss / (idx + 1), 100 * train_acc / len(train_dataloader.dataset),
            train_acc, len(train_dataloader.dataset), optimizer.param_groups[0]['lr']))
        if WANDB:
            wandb.log({"epoch": epoch, "lr": optimizer.param_groups[0]['lr']})
        model.eval()
        test_acc = 0
        test_loss = 0.
        tot_count = 0
        with torch.no_grad():
            for idx, (data, target) in enumerate(tqdm(val_dataloader)):
                data, target = data.to(args.device), target.to(args.device)
                pred = model(data)
                loss = corss_loss(pred, target)
                test_loss += loss.item()
                test_acc += pred.argmax(dim=1).eq(target.argmax(dim=1)).sum().item()
                tot_count += target.shape[0]

        print('test Loss:{:.3f} Acc: {:.2f}% {}/{}'.format(
            test_loss / (idx + 1), 100 * test_acc / len(val_dataloader.dataset),
            test_acc, len(val_dataloader.dataset)))
        wandb.log({"val_acc": test_acc / len(val_dataloader.dataset)})

        test_acc /= len(val_dataloader.dataset)

        if test_acc > best_acc:
            print('Save ...')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(),
                       './checkpoint/fasttext/fasttext_model_{:.2f}_epoch_{}.pth'.format(100 * test_acc, epoch))
            best_acc = test_acc


if __name__ == '__main__':
    train()
