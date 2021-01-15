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
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Head Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--resume', default='', help='resume from checkpoint', action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", default="seg_head.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='./logs', help='path where to save output models and logs')

    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, num_classes):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            feature, label, index = data['feature'].to(device), data['label'].to(device), data['index']
            output = model(feature)
            output = output.argmax(1)
            confmat.update(label.cpu().flatten(), output.cpu().flatten())
            visual2d(output.cpu()[0], index[0]) 
        confmat.reduce_from_all_processes()
    return confmat

def main(args):

    with open("configs/config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Config file Read successfully!")
        print(cfg)

    validation_split = cfg['val_split']
    shuffle_dataset = True
    random_seed= cfg['seed']
    num_classes = cfg['num_classes']
    grid_size = cfg['grid_size']

    writer = SummaryWriter(args.save_dir)

    dataset = FeaturesDataset(feat_dir='/home/josh94mur/data/features', label_dir='/home/josh94mur/data/targets/')
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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], 
                                                sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'],
                                                sampler=valid_sampler)
    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['feature'].size(), sample_batched['label'].size())
    
    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    if args.test_only:
        print("-----------------------------------------")
        print("Use : tensorboard --logdir logs/eval_data ")
        model = Seg_Head()
        model.to(device)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
        confmat = evaluate(model, valid_loader, device=device, num_classes=num_classes)
        print(confmat)
        print("Finished Testing!")
        return
    else:
        print("-----------------------------------------")
        print("Use : tensorboard --logdir logs ")
        model = Seg_Head()
        model.to(device)

        if args.resume:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            model.load_state_dict(checkpoint)

        criterion = FocalLoss(gamma=2, reduction='mean')
        optimizer = optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

        model.train()
        num_epochs = cfg['epochs'] 
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
                    writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('Learning rate', optimizer.param_groups[0]["lr"], epoch * len(train_loader) + i)
                    sleep(0.01)

            confmat = evaluate(model, valid_loader, device=device, num_classes=num_classes)

            writer.add_scalar(f'accuracy', confmat.acc_global, epoch)
            writer.add_scalar(f'mean_IoU', confmat.mean_IoU, epoch)

        PATH = 'seg_head1.pth'
        torch.save(model.state_dict(), PATH)
        print('Finished Training. Model Saved!')

if __name__=="__main__":
    args = parse_args()
    main(args)

# TODO : add Parallel training support
# TODO : move to pytrorch lighting!
# TODO : fix gpu_id == 1 :)   changing lr + model graph 




