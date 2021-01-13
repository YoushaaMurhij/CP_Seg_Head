"""
-----------------------------------------------------------------------------------
# Author: Youshaa Murhij
# DoC: 2021.01.5
# email: yosha.morheg@gmail.com
-----------------------------------------------------------------------------------
# Description: Training script for Semantic Head
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import FeaturesDataset
from model import Seg_Head
from visualizer import visual2d
from loss import FocalLoss
from utils import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from time import sleep
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Head Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of epochs')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--resume', default='', help='resume from checkpoint', action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", default="seg_head.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./data/logs', help='path where to save output models and logs')

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for data in dataloader:
            feature, label, index = data['feature'].to(device), data['label'].to(device), data['index']
            output = model(feature)
            output = output.argmax(1)
            confmat.update(label.cpu().flatten(), output.cpu().flatten())
            #visual2d(output.cpu()[0], index[0])  
        confmat.reduce_from_all_processes()
    return confmat

def main(args):

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    num_classes = 33
    grid_size = 256

    writer = SummaryWriter(args.save_dir)

    dataset = FeaturesDataset(feat_dir='./data/features', label_dir='./data/targets/')
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                                sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=valid_sampler)
    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['feature'].size(), sample_batched['label'].size())
    
    device = torch.device(args.device)

    if args.test_only:
        model = Seg_Head()
        model.to(device)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                feature, label, index = data['feature'].to(device), data['label'].to(device), data['index']
                output = model(feature)
                output = output.argmax(1)
                visual2d(output.cpu()[0], index[0]) 
        confmat = evaluate(model, valid_loader, device=device, num_classes=num_classes)
        print(confmat)
        # print("Finished Testing!")
        return
    else:
        model = Seg_Head()
        model.to(device)

        if args.resume:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            model.load_state_dict(checkpoint)

        criterion = FocalLoss(gamma=2, reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        model.train()
        num_epochs = args.epochs 
        for epoch in range(num_epochs):
              
            with tqdm(train_loader, unit = "batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    features = data['feature']
                    labels = data['label']
                    if torch.cuda.is_available(): 
                        labels = labels.cuda()

                    optimizer.zero_grad()
                    outputs = model(features)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    tepoch.set_postfix(loss=loss.item())
                    writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch * len(data_loader) + i)
                    sleep(0.01)

            confmat = evaluate(model, valid_loader, device=device, num_classes=num_classes)
            #print(confmat)
            #print(f'accuracy={confmat.acc_global}, mean_IoU={confmat.mean_IoU}')
            for classname, iu in iou.items():
                writer.add_scalar(f'{classname} IoU', iu, current_epoch)
                
        PATH = args.save_dir+'/seg_head.pth'
        torch.save(model.state_dict(), PATH)
        print('Finished Training. Model Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

# TODO : add tensorboard support && config
# TODO : add Parallel training support
# TODO : move to pytrorch lighting!
# TODO : add accuracey metric (from sem seg)
# TODO : fix gpu_id == 1 :)