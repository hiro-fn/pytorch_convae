import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision.datasets as datasets
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os

from net import AENet

image_size = 128

def create_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    return  transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def set_dataset(dataset_path, transform, options):
    batch_size = options['batch_size']
    num_worker = options['num_workers']
    my_datasets = datasets.ImageFolder(root=dataset_path, transform=transform)

    return torch.utils.data.DataLoader(my_datasets, batch_size=batch_size)

def to_image(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, image_size, image_size)
    return x


def run_train(model, datatset, options):    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'],
                                weight_decay=1e-5)

    max_epoch = options['epoch']
    for epoch in range(max_epoch):
        for data in datatset:
            raw_images, _ = data
            image = raw_images.to('cuda')

            output = model(image)
            loss = criterion(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'[{epoch+1}/{max_epoch}]: loss:{loss.data[0]}')
        if epoch % 10 == 0:
            pic = to_image(output.cpu().data)
            save_image(pic, f'./image/image_{epoch}.png')

    torch.save(model.state_dict(), './conv_autoencoder.pth')

def main():
    train_dataset_path = "D:\\project\\idol_classification\\images\\train\\"

    options = {
        'batch_size': 128,
        'epoch': 300,
        'lr': 1e-3,
        'num_workers': 4,
        'decay': 1e-5
    }

    data_transform = create_transform()
    train_dataset = set_dataset(train_dataset_path, data_transform, options)
    print(type(train_dataset))
    print(type(data_transform))
    #return

    model = AENet().to('cuda')
    run_train(model, train_dataset, options)

if __name__ == '__main__':
    main()