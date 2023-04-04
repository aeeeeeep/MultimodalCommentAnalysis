import random
from collections import Counter, OrderedDict

import torch
import torch.nn as nn
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler
from torchtext.data.utils import ngrams_iterator
from torchtext.transforms import VocabTransform
from torchtext.vocab import vocab

from utils import *


class Args:
    def __init__(self) -> None:
        self.batch_size = 64
        self.lr = 0.5
        self.epochs = 12
        self.radio = 0.7
        self.num_workers = 12
        self.full_list = False

        self.embed_size = 100
        self.hidden_size = 16
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
    def __init__(self, df, df_img, flag='train', max_length=20) -> None:
        df.set_index('GlobalID')
        df_img.set_index('GlobalID')
        df.drop(df[df['Notes'] == ' '].index, inplace=True)
        df = pd.concat([df[df['Lab Status'] == 'Negative ID'], df[df['Lab Status'] == 'Positive ID']])
        df['label'] = df['Lab Status'].apply(lambda i: 0 if i == 'Negative ID' else 1)
        df['Notes'] = df['Notes'].apply(str)
        df.drop_duplicates(subset=['Notes'], inplace=True)
        df['Notes_token'] = process_notes(df['Notes'])
        df.drop(df[df['Notes_token'].apply(lambda x: len(x) < 3)].index, inplace=True)
        data = df[['Notes', 'label']]
        self.text_list = data['Notes']
        self.flag = flag
        self.max_length = max_length
        assert self.flag in ['train', 'val'], 'not implement!'
        train_data, val_data = data_split(data, ratio=args.radio, shuffle=True)
        if self.flag == 'train':
            self.text_vocab, self.vocab_transform = self.reform_vocab(train_data['Notes'].to_list())
            self.text_label = train_data['label'].to_list()
            self.fast_data = self.generate_fast_text_data()
            self.len = len(train_data)

        else:
            self.text_vocab, self.vocab_transform = self.reform_vocab(val_data['Notes'].to_list())
            self.text_label = val_data['label'].to_list()
            self.fast_data = self.generate_fast_text_data()
            self.len = len(val_data)

    def __getitem__(self, index):
        data_row = self.fast_data[index]
        data_row = pad_or_cut(data_row, self.max_length)
        data_label = torch.zeros(2, dtype=torch.float32)
        data_label[self.text_label[index]] = 1
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
    df = pd.read_excel('../2021_MCM_Problem_C_Data/2021MCMProblemC_DataSet.xlsx')
    df_img = pd.read_excel('../2021_MCM_Problem_C_Data/2021MCM_ProblemC_Images_by_GlobalID.xlsx')
    train_dataset = Dataset(df=df, df_img=df_img, flag='train')
    train_dataloader = DataLoaderX(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                   shuffle=False, sampler=ImbalancedDatasetSampler(train_dataset))
    val_dataset = Dataset(df=df, df_img=df_img, flag='val')
    val_dataloader = DataLoaderX(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, drop_last=True)

    model = Net(train_dataset.get_vocab_size()).to(args.device)
    corss_loss = nn.CrossEntropyLoss()
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
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            train_acc += pred.argmax(dim=1).eq(target.argmax(dim=1)).sum().item()
            tot_count += target.shape[0]

        print('train Loss:{:.3f} Acc: {:.2f}% {}/{} lr:{:.2e}'.format(
            train_loss / (idx + 1), 100 * train_acc / len(train_dataloader.dataset),
            train_acc, len(train_dataloader.dataset), optimizer.param_groups[0]['lr']))
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

        test_acc /= len(val_dataloader.dataset)

        if test_acc > best_acc:
            print('Save ...')
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(model.state_dict(),
                       './checkpoint/fasttext_model_{:.2f}_epoch_{}.pth'.format(100 * test_acc, epoch))
            best_acc = test_acc


if __name__ == '__main__':
    train()
