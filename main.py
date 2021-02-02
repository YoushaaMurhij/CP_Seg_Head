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
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import FeaturesDataset
from model import Seg_Head, UNET, New_Head, get_model
from visualizer import visual2d
from loss import *
from losses import GDiceLossV2
from utils import *
from tqdm import tqdm
import numpy as np
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import yaml
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Head Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--resume', default='', help='resume from checkpoint', action="store_true")
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--pretrained", default="seg_head.pth", help="Use pre-trained models")
    parser.add_argument('--save_dir', default='/logs/train_data/', help='path where to save output models and logs')
    parser.add_argument('--focal_loss', action='store_true', help='train with focal loss')
    args = parser.parse_args()
    return args

def evaluate(model, dataloader, device, num_classes, save_dir, criterion=None, epoch=None, writer=None):
    model.eval()
    confmat = ConfusionMatrix(num_classes)
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            feature, label, index = data['feature'].to(device), data['label'].to(device), data['index']
            output = model(feature)

            if criterion is not None:
                loss = criterion(output, label)
                writer.add_scalar('Validation Loss', loss.item(), epoch * len(dataloader) + i)
            output = output.argmax(1)
            confmat.update(label.cpu().flatten(), output.cpu().flatten())

            visual2d(output.cpu()[0], index[0], save_dir, str(epoch))
            # if writer is not None:
            #     img_grid = torch.reshape(output, (-1, 1, 256, 256))
            #     writer.add_image('Evaluattion point cloud grids:', img_grid, dataformats='NCHW')
        confmat.reduce_from_all_processes()
    return confmat

def main(args):

    with open("configs/config.yaml", "r") as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Config file Read successfully!")
        print(cfg)
    
    now = datetime.now()
    tag = " -  5 * conv2d + interpolation - 25 epoch + dropout 0.1 (256-128-64-32) + kernel (1-1-1-1-1) - FL sad weights !"
    save_str = '.' + args.save_dir + now.strftime("%d-%m-%Y-%H:%M:%S") + tag
    print("------------------------------------------")
    print("Use : tensorboard --logdir logs/train_data")
    print("------------------------------------------")

    num_epochs = cfg['epochs'] 
    validation_split = cfg['val_split']
    shuffle_dataset = True
    random_seed = cfg['seed']
    input_size = cfg['input_size']
    num_classes = cfg['num_classes']
    grid_size = cfg['grid_size']
    learning_rate = cfg['lr']
    batch_size = cfg['batch_size']
    momentum = cfg['momentum']
    weight_decay = cfg['weight_decay']

    writer = SummaryWriter(save_str)

    device = torch.device(args.device)
    print(f'cuda device is: {device}')

    dataset = FeaturesDataset(feat_dir='/home/josh94mur/data/features', label_dir='/home/josh94mur/data/targets/', device=device) 
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    test_indices = indices[:200]
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(dataset, batch_size=batch_size // 8, sampler=valid_sampler)
    test_loader = DataLoader(dataset, batch_size=1, sampler=test_sampler)

    model = Seg_Head().to(device)

    if args.test_only:
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        model.load_state_dict(checkpoint)
        confmat = evaluate(model, test_loader, device, num_classes, save_str)
        print(confmat)
        print("Finished Testing!")
        return
    else:
        if args.resume:
            checkpoint = torch.load(args.pretrained, map_location='cpu')
            model.load_state_dict(checkpoint)
        writer.add_graph(model, torch.randn(1, 384, 128, 128, requires_grad=False).to(device))

        if args.focal_loss:
            criterion = FocalLoss_(gamma=2, alpha=CLASS_WEIGHTS)
        else:
            loss_weights = torch.FloatTensor(CLASS_WEIGHTS).to(device)
            criterion = nn.CrossEntropyLoss(weight=loss_weights)

        optimizer = optim.SGD(model.parameters(), weight_decay = weight_decay, lr=learning_rate, momentum=momentum)
        #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=learning_rate)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / (len(train_loader) * num_epochs)) ** 0.8)

        model.train()
        for epoch in range(num_epochs):
              
            with tqdm(train_loader, unit = "batch") as tepoch:
                for i, data in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch}")

                    features = data['feature']
                    labels = data['label']
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(features)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    tepoch.set_postfix(loss=loss.item())
                    writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + i)
                    writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], epoch * len(train_loader) + i) #optimizer.param_groups[0]['lr']
                    sleep(0.01)

            confmat = evaluate(model, valid_loader, device=device, num_classes=num_classes, save_dir=save_str, criterion=criterion, epoch=epoch, writer=writer)
            print(confmat)
            writer.add_scalar(f'accuracy', confmat.acc_global, epoch)
            writer.add_scalar(f'mean_IoU', confmat.mean_IoU, epoch)

            PATH = save_str +'/seg_head_Epoch_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), PATH)
        print('Finished Training. Model Saved!')
        writer.close()

if __name__=="__main__":
    args = parse_args()
    main(args)

# TODO : reduce # classes if applicable 
# TODO : move to pytorch lighting!
# TODO :
# TODO :  
# TODO : expand the dataset [03]
