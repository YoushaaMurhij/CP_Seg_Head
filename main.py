import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import *
from model import *
from loss import FocalLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Head Training')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.007, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    args = parser.parse_args()
    return args

def main(args):

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

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

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                sampler=valid_sampler)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['feature'].size(), sample_batched['label'].size())
    
    device = args.device
    model = Seg_Head()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    #criterion = FocalLoss(gamma=2)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    num_epochs = args.epochs # loop over the dataset multiple times
    for epoch in range(num_epochs):  

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            features = data['feature']
            labels = data['label']
            if torch.cuda.is_available(): 
                labels = labels.cuda()

            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 4 == 3:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 4))
                running_loss = 0.0

    print('Finished Training')

    PATH = './seg_head.pth'
    torch.save(net.state_dict(), PATH)

if __name__=="__main__":
    args = parse_args()
    main(args)
