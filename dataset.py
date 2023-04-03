import time
import traceback

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer


class BaseDataset(Dataset):
    def _try_getitem(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        wait = 0.1
        while True:
            try:
                ret = self._try_getitem(idx)
                return ret
            except KeyboardInterrupt:
                break
            except (Exception, BaseException) as e:
                exstr = traceback.format_exc()
                print(exstr)
                print('read error, waiting:', wait)
                time.sleep(wait)
                wait = min(wait * 2, 1000)


class BartDataset(BaseDataset):
    def __init__(self, data_file, sos_id=0, eos_id=2, pad_id=1):
        self.df = pd.read_csv(data_file)
        self.sos_id = sos_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.tokenizer = BartTokenizer.from_pretrained('/root/autodl-tmp/pretrain/')
        self.text = self.df['text'].values
        self.labels = self.df['label'].values

    def __len__(self):
        return len(self.df)

    def _try_getitem(self, idx):
        source = self.text[idx]
        source_ids = self.tokenizer(source, max_length=512, padding='max_length', truncation=True)
        target = torch.zeros(2, dtype=torch.float32)
        try:
            target[int(self.labels[idx])] = 1.0
        except:
            return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(source_ids['attention_mask'])
        return torch.LongTensor(source_ids['input_ids']), torch.LongTensor(source_ids['attention_mask']), target
