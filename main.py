import copy
import os
import argparse
from copy import deepcopy
from datetime import datetime

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from models.EEGNet import EEGNet


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(exp_name, train_acc_list, epochs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, train_acc_list, marker='.')
    plt.savefig(f"InferenceResult/{exp_name}/train_acc.png")
    
def plot_train_loss(exp_name, train_loss_list, epochs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, train_loss_list, marker='.')
    plt.savefig(f"InferenceResult/{exp_name}/train_loss.png")

def plot_test_acc(exp_name, test_acc_list, epochs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, test_acc_list, marker='.')
    plt.savefig(f"InferenceResult/{exp_name}/test_acc.png")

def plot_lr(exp_name, lr_list, epochs):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(epochs, lr_list, marker='.')
    plt.savefig(f"InferenceResult/{exp_name}/lr.png")

def train(exp_name, model, loader, test_loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    lr_list = []
    
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=100, verbose=True)
    # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=30, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.9, patience=50, verbose=True)
    for epoch in range(1, args.num_epochs+1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch} Loss: {avg_loss} Training Acc. (%): {avg_acc:3.2f}%')
        
        current_lr = optimizer.param_groups[0]['lr']
        lr_list.append(current_lr)

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            print(f"Update best model weight at {epoch} from {best_acc} to {test_acc}")
            best_acc = test_acc
            best_wts = model.state_dict()
            os.makedirs(f'./weights/{exp_name}', exist_ok=True)
            torch.save(best_wts, f'./weights/{exp_name}/best.pt')

        scheduler.step(test_acc)
        
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    return avg_acc_list, avg_loss_list, test_acc_list, lr_list


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_epochs", type=int, default=1500)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=0.001)
    args = parser.parse_args()

    current_time = datetime.now()
    exp_name = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # exp_name = "2024-04-23 14:27:31"
    os.makedirs('InferenceResult/', exist_ok=True)
    os.makedirs(f'InferenceResult/{exp_name}', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EEGNet()

    print(model)
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list, lr_list = train(exp_name, model, train_loader, test_loader, criterion, optimizer, args)

    plot_train_acc(exp_name, train_acc_list, list(range(args.num_epochs)))
    plot_train_loss(exp_name, train_loss_list, list(range(args.num_epochs)))
    plot_test_acc(exp_name, test_acc_list, list(range(args.num_epochs)))
    plot_lr(exp_name, lr_list, list(range(args.num_epochs)))

    print("testing")
    test_model = EEGNet()
    test_model.to(device)
    test_model.load_state_dict(torch.load(os.path.join("weights", exp_name, "best.pt")))
    test_acc = test(test_model, test_loader)
    print(f"testing acc: {test_acc:.2f}")