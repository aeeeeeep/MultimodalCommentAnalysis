import os
import random
import shutil

book_dir = "../data/img_dataset/book/"
nobook_dir = "../data/img_dataset/nobook/"

train_dir = "../data/img_dataset/train/"
test_dir = "../data/img_dataset/val/"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_ratio = 0.7
test_ratio = 0.3

for class_dir in [book_dir, nobook_dir]:
    file_list = os.listdir(class_dir)
    random.shuffle(file_list)
    num_train = int(len(file_list) * train_ratio)
    train_files = file_list[:1750]
    test_files = file_list[-750:]
    for file_name in train_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(train_dir, file_name)
        shutil.copy(src_path, dst_path)
    for file_name in test_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(test_dir, file_name)
        shutil.copy(src_path, dst_path)
