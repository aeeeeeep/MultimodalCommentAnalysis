import csv
import random

pos_samples = []
neg_samples = []

with open('../data/label.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:  # Skip header
            continue
        sample = {'ID': row[0], 'text': row[1], 'vote': row[2], 'label': row[3]}
        if sample['label'] == '1.0':
            pos_samples.append(sample)
        else:
            neg_samples.append(sample)

random.shuffle(pos_samples)
random.shuffle(neg_samples)

num_pos_train = int(len(pos_samples) * 0.7)
num_neg_train = int(len(neg_samples) * 0.7)

train_samples = pos_samples[:10000] + neg_samples[:10000]
val_samples = pos_samples[2500:] + neg_samples[2500:]

random.shuffle(train_samples)
random.shuffle(val_samples)

with open('../data/train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'text', 'label'])
    for sample in train_samples:
        writer.writerow([sample['ID'], sample['text'], sample['label']])

with open('../data/val.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['ID', 'text', 'label'])
    for sample in val_samples:
        writer.writerow([sample['ID'], sample['text'], sample['label']])

