import os
import random
import shutil

# set the path of the directories containing the images
book_dir = "../data/img_dataset/book/"
nobook_dir = "../data/img_dataset/nobook/"

# set the path of the directories to store the training and test sets
train_dir = "../data/img_dataset/train/"
test_dir = "../data/img_dataset/val/"

# create the directories for the training and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# set the ratio of training and test images
train_ratio = 0.7
test_ratio = 0.3

# loop through each class directory and divide the images into training and test sets
for class_dir in [book_dir, nobook_dir]:
    # get the list of image files in the class directory
    file_list = os.listdir(class_dir)
    # shuffle the list of image files
    random.shuffle(file_list)
    # calculate the number of training images based on the ratio
    num_train = int(len(file_list) * train_ratio)
    # split the list of image files into training and test sets
    train_files = file_list[:1750]
    test_files = file_list[-750:]
    # copy the training images to the training directory
    for file_name in train_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(train_dir, file_name)
        shutil.copy(src_path, dst_path)
    # copy the test images to the test directory
    for file_name in test_files:
        src_path = os.path.join(class_dir, file_name)
        dst_path = os.path.join(test_dir, file_name)
        shutil.copy(src_path, dst_path)
