import json
import os
import random
import pandas as pd
from io import BytesIO
from functools import partial

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer

def create_dataloaders(args):
    train_dataset = MultiModalDataset(args, args.train_data_file)
    val_dataset = MultiModalDataset(args, args.val_data_file)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers, prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)

    train_sampler = RandomSampler(train_dataset)
    val_sampler = SequentialSampler(val_dataset)
    train_dataloader = dataloader_class(train_dataset,
                                        batch_size=args.batch_size,
                                        sampler=train_sampler,
                                        drop_last=True)
    val_dataloader = dataloader_class(val_dataset,
                                      batch_size=args.val_batch_size,
                                      sampler=val_sampler,
                                      drop_last=False)
    return train_dataloader, val_dataloader

class MultiModalDataset(Dataset):
    """ A simple class that supports multi-modal inputs.
    For the visual features, this dataset class will read the pre-extracted
    features from the .npy files. For the text information, it
    uses the BERT tokenizer to tokenize. We simply ignore the ASR & OCR text in this implementation.
    Args:
        ann_path (str): annotation file path, with the '.json' suffix.
        zip_feats (str): visual feature zip file path.
        test_mode (bool): if it's for testing.
    """

    def __init__(self,
                 args,
                 data_file: str,
                 test_mode: bool = False):
        self.bert_seq_length = args.bert_seq_length
        self.test_mode = test_mode
        self.df = pd.read_csv(data_file)
        self.id = self.df['ID'].values
        self.text = self.df['text'].values
        self.labels = self.df['label'].values
        self.img_list = []
        for i in os.listdir("../data/Books_5_images"):
                self.img_list.append(i[:-4])
        self.img_list = set(self.img_list)
        # initialize the text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_dir, use_fast=True, cache_dir=args.bert_cache)
        self.transform = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.id)

    def get_visual_feats(self, idx: int) -> tuple:
        img = Image.open('./data/Books_5_images/' + self.id[idx] + '.jpg')
        img = img.convert("RGB")
        img_tensor = self.transform(img)
        mask = torch.ones((1,), dtype=torch.long)
        return img_tensor, mask

    def tokenize_text(self, text: str) -> tuple:
        words = text.split()
        drop_words = " ".join(words[:128] + words[-128:])
        encoded_inputs = self.tokenizer(drop_words, max_length=self.bert_seq_length, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx: int) -> dict:
        # Step 1, load visual features from zipfile.
        image_input, image_mask = self.get_visual_feats(idx)

        # Step 2, load text tokens
        text_input, text_mask = self.tokenize_text(self.text[idx])

        # Step 3, summarize into a dictionary
        data = dict(
            image_input=image_input,
            image_mask=image_mask,
            text_input=text_input,
            text_mask=text_mask
        )

        # Step 4, load label if not test mode
        if not self.test_mode:
            label = torch.zeros(2, dtype=torch.float32)
            label[int(self.labels[idx])] = 1.0
            data['label'] = label

        return data
