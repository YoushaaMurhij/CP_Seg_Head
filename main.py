import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import *
from model import *

def main():

    batch_size = 1
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    dataset = FeaturesDataset(feat_dir='./data/features', label_dir='./data/targets/')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

    # for i_batch, sample_batched in enumerate(train_loader):
    #     print(i_batch, sample_batched['feature'].size(), sample_batched['label'].size())
    
    device = 'cuda'
    model = Seg_Head()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 2 # loop over the dataset multiple times
    for epoch in range(num_epochs):  

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            features = data['feature']
            labels = data['label']
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(features)
            print(labels.shape)
            print(outputs.shape)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './seg_head.pth'
    torch.save(net.state_dict(), PATH)

if __name__=="__main__":
    main()