import os
import gc
import torch
import shutil
import tempfile
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
from sklearn.metrics import classification_report
from torchvision import datasets, transforms, models

data_dir = './data/img_dataset'
class_names0 = os.listdir(data_dir)

class_names=[]
for item in class_names0:
    class_names+=[item]
num_class = len(class_names)
image_files = [[os.path.join(data_dir, class_name, x) \
                for x in os.listdir(os.path.join(data_dir, class_name))[:5000]] \
                for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
num_total = len(image_label_list)

width, height = Image.open(image_file_list[0]).size

print("Total image count:", num_total)
print("Image dimensions:", width, "x", height)
print("Label names:", class_names)
print("Label counts:", [len(image_files[i]) for i in range(num_class)])

valid_frac = 0.3
trainX, trainY = [], []
valX, valY = [], []
testX, testY = [], []

if not os.path.exists('./book_train'):
    print("Create dataset...")
    for i in tqdm(range(num_total)):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./book_val/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./book_val/{k}/', r))
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])
            j = image_file_list[i]
            k = j.split('/')[-2]
            r = j.split('/')[-1]
            os.makedirs(f'./book_train/{k}', exist_ok=True)
            shutil.copyfile(j, os.path.join(f'./book_train/{k}/', r))
else:
    trainX = os.listdir('./book_train')
    valX = os.listdir('./book_val')
          
print(len(trainX), len(valX))

train_dir = './book_train'
val_dir = './book_val'
batch_size = 64

transform_train = transforms.Compose(
    [transforms.ToTensor(),
     transforms.RandomAffine(degrees=(10, 150), translate=(0.2, 0.5), shear=45),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([224, 224]),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
     ])

transform_val = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize([224, 224]),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
     ])

image_datasets = {}
image_datasets["train"] = datasets.ImageFolder(root = train_dir, transform=transform_train)
image_datasets["valid"] = datasets.ImageFolder(root = val_dir, transform=transform_val)

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

print(class_names)

train_loader = torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle = True,
                                          num_workers = 8)
valid_loader = torch.utils.data.DataLoader(image_datasets["valid"], batch_size=batch_size, shuffle = False,
                                          num_workers = 8)

print(dataset_sizes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


parser = argparse.ArgumentParser()

parser.add_argument('-save_dir', action="store", dest="save_dir", type=str, default="./checkpoint/cnn/")
parser.add_argument('-lr', action="store", dest="lr", type=float, default=1e-4)
parser.add_argument('-hiddenunits', action="store", dest="hiddenunits", type=int, default=128)
parser.add_argument('-epochs', action="store", dest="epochs", type=int, default=3)
ins=parser.parse_args(args=[])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = tv.models.convnext_tiny(weights=tv.models.ConvNeXt_Tiny_Weights)
        self.backbone = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
        self.fc1 = nn.Linear(1000, ins.hiddenunits)
        self.fc2 = nn.Linear(ins.hiddenunits, 2)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x

        
model = Model()
print("Model:\n", model)

gc.collect()
torch.cuda.empty_cache()

use_gpu = torch.cuda.is_available()
print("Use GPU: ", use_gpu)

steps = int(len(trainX) / batch_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = ins.lr)

model.cuda()

train_loss=[]
test_loss=[]
train_accuracy=[]
test_accuracy=[]
best_acc = 0

print("Training...")
# Training
for epoch in range(ins.epochs):
    # Reset variables at 0 epoch
    correct=0
    iteration=0
    iter_loss=0.0
  
    model.train() # training mode

    with tqdm(total=len(train_loader)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch, ins.epochs - 1))
        for i,(inputs,labels) in enumerate(train_loader):
            inputs=Variable(inputs)
            labels=Variable(labels)
            cuda=torch.cuda.is_available()
            if cuda:
                inputs=inputs.cuda()
                labels=labels.cuda()
            optimizer.zero_grad() # clear gradient
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            iter_loss += loss.item() # accumulate loss
            loss.requires_grad_(True)
            loss.backward() # backpropagation
            optimizer.step() # update weights

            # save the correct predictions for training data
            _,predicted=torch.max(outputs,1)
            accuracy = (predicted.cpu()==labels.cpu()).sum()
            correct += accuracy
            iteration +=1
            
            _tqdm.set_postfix(loss='{:.3f}'.format(loss), accuracy='{:.3f}'.format(accuracy / len(labels)),
                    lr='{:.1e}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
            _tqdm.update(1)

    train_loss.append(iter_loss/iteration)
    train_accuracy.append((100*correct/len(image_datasets["train"])))
  
    correct=0
    iteration=0
    valid_loss=0.0
    print("testing...")
  
    model.eval()  
  
    with tqdm(total=len(valid_loader)) as _tqdm:
        for i, (inputs, labels) in enumerate(valid_loader):
            inputs=Variable(inputs)
            labels=Variable(labels)

            cuda=torch.cuda.is_available()
            if cuda:
                inputs=inputs.cuda()
                labels=labels.cuda()
            with torch.no_grad():
                outputs=model(inputs)
                loss=criterion(outputs,labels)
                valid_loss += loss.item()

                _,predicted=torch.max(outputs,1)
                correct+=(predicted.cpu()==labels.cpu()).sum()

                iteration+=1  
            
            _tqdm.update(1)
    
    test_loss.append(valid_loss/iteration)
    test_accuracy.append((100*correct/len(image_datasets["valid"])))
    
    print('epoch {}/{}, training loss:{:.3f}, training accuracy:{:.3f}, testing loss {:.3f}, testing accuracy:{:.3f}'
       .format(epoch+1, ins.epochs, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))
    state_dict = model.module.state_dict() if next(model.parameters()).device == 'cuda:0' else model.state_dict()
    torch.save({'epoch': epoch, 'model_state_dict': state_dict},
            f'./{ins.save_dir}/model_epoch_{epoch+1}.pth')
