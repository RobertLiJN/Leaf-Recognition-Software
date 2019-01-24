
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from time import time
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(20*53*53, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 20*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.squeeze(1)
        # x = nn.functional.log_softmax(x, dim=1)
        return x


def get_name(args):
    name = 'batch size {} learning rate {} epochs {} model {} transfer {} valid ratio {} shuffle {} seed {}'.format(
        args.batch_size, args.learning_rate, args.epochs, args.model, args.transfer,
        args.valid_ratio, args.shuffle, args.seed
    )

    return name


def get_loader(args):
    valid_ratio = args.valid_ratio
    batch_size = args.batch_size
    shuffle = args.shuffle
    seed = args.seed

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    data_dir = os.getcwd() + '/data_binary'
    train_datasets = datasets.ImageFolder(data_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(data_dir, transform=valid_transforms)

    length = len(train_datasets)
    indices = list(range(length))
    split = int(np.floor(valid_ratio * length))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_ind, valid_ind = indices[split:], indices[:split]
    train_sam = SubsetRandomSampler(train_ind)
    valid_sam = SubsetRandomSampler(valid_ind)

    train_loader = DataLoader(train_datasets, batch_size=batch_size, sampler=train_sam)
    valid_loader = DataLoader(valid_datasets, batch_size=batch_size, sampler=valid_sam)

    return train_loader, valid_loader


def write_record(record, args):
    record = np.transpose(record)

    df = pd.DataFrame({
        'step': record[0], 'time': record[1], 'epoch': record[2], 'train_loss': record[3],
        'train_error': record[4], 'valid_error': record[5]
    })
    df.to_csv(os.getcwd() + '/result/{}.csv'.format(get_name(args)), index=False)

    return


def evaluate(model, valid_loader):
    correct = 0

    for i, data in enumerate(valid_loader, 1):
        inputs, label = data[0], data[1]
        outputs = model(inputs)
        _, predict = torch.max(outputs, 1)
        correct += torch.sum(predict == label)

    return correct.double() / len(valid_loader.dataset)


def load_model(args):

    learning_rate = args.learning_rate
    model = Net()

    # optimizer = optim.Adam(model.fc.parameters() if transfer else model.parameters(), lr=learning_rate, eps=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    loss_fnc = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, loss_fnc


def train(args, train_loader, valid_loader):
    epochs = args.epochs
    curr_path = os.getcwd()

    if not os.path.exists(curr_path + '/model'):
        os.makedirs(curr_path + '/model')

    if not os.path.exists(curr_path + '/result'):
        os.makedirs(curr_path + '/result')

    model, optimizer, scheduler, loss_fnc = load_model(args)

    steps = 0
    record = []
    max_acc = 0
    start = time()

    for epoch in range(epochs):
        # scheduler.step()

        running_loss = 0.0
        running_correct = 0

        for i, data in enumerate(train_loader, 1):

            steps += 1
            print(steps)
            inputs, label = data[0], data[1]

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predict = torch.max(outputs, 1)
            loss = loss_fnc(outputs, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(predict == label.data)

        eval_acc = evaluate(model, valid_loader)

        # if eval_acc >= max_acc:
        #     max_acc = eval_acc
        #     torch.save(model, curr_path + '/model/{}.pt'.format(get_name(args)))

        rec = [
            steps, time() - start, epoch + 1, running_loss/(i+1), running_correct.double()/ len(train_loader.dataset),
            eval_acc
        ]
        # record.append(rec)

        print('ep:', rec[2], ', los:', rec[3], ', acc:', rec[4], ', val:', rec[5])

    # write_record(record, args)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--valid_ratio', type=float, default=0.2, help='validation set proportion')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--transfer', type=int, default=1, help='transfer learning')
    parser.add_argument('--model', type=str, default='resnet', help='type of model')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--shuffle', type=int, default=1, help='shuffle or not')
    parser.add_argument('--epochs', type=int, default=25, help='num of epochs')
    parser.add_argument('--seed', type=int, default=324, help='random seed')

    args = parser.parse_args()

    train_loader, valid_loader = get_loader(args)

    train(args, train_loader, valid_loader)
