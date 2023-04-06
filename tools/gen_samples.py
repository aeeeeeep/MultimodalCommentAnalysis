import os
import csv
import random
from tqdm import tqdm

samples_0 = []
samples_1 = []
samples_2 = []

img_list = []
for i in os.listdir("../data/Books_5_images"):
    img_list.append(i[:-4])
img_list = set(img_list)

with open('../data/label.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for i, row in tqdm(enumerate(reader)):
        if i == 0: 
            continue
        sample = {'ID': row[0], 'text': row[1], 'label': row[2]}
        if sample['ID'] in img_list and sample['label'] == '2.0':
            sample['label'] = "1.0"
            samples_2.append(sample)
        elif sample['ID'] in img_list and sample['label'] == '0.0':
            samples_0.append(sample)
        # elif sample['ID'] in img_list and sample['label'] == '1.0':
        #     samples_1.append(sample)

print(len(samples_0))
# print(len(samples_1))
print(len(samples_2))
exit()

random.shuffle(samples_0)
# random.shuffle(samples_1)
random.shuffle(samples_2)

num_0_train = int(len(samples_0) * 0.7)
# num_1_train = int(len(samples_1) * 0.7)
num_2_train = int(len(samples_2) * 0.7)

train_samples = samples_0[:10000] + samples_2[:10000]
# train_samples = samples_0[:num_0_train] + samples_1[:num_1_train] + samples_2[:num_2_train]
val_samples = samples_0[-3000:] + samples_2[-3000:]
# val_samples = samples_0[-num_0_train:] + samples_1[-num_1_train:] + samples_2[-num_2_train:]

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
