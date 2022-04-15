from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from pathlib import Path
import torch
path = Path('.')

def add_arrow(img):
    '''
    Method to add arrow to image
    :param img: raw image
    :return: arrowed image
    '''
    start = 2
    for i in range(start,start+7):
        img[:,i,start+6] = -1
    for i in range(start+5,start+8):
        img[:,start+1,i] = -1
    for i in range(start+4,start+9):
        img[:,start+2,i] = -1
    return img

class arrowedCIFAR(Dataset):
    def __init__(self, train=True, clean_data=False):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train = train
        if self.train:
            self.cifar = torchvision.datasets.CIFAR10(root=path / 'data', download=True, transform=transform,
                                                      train=True)
        else:
            self.cifar = torchvision.datasets.CIFAR10(root=path / 'data', download=True, transform=transform,
                                                      train=False)
        self.data = []
        self.labels = []
        for i in tqdm(range(len(self.cifar))):
            img, orig_label = self.cifar.__getitem__(i)
            if not clean_data:
                img = add_arrow(img)
            self.data.append(img)  # Only care about the rotation
            self.labels.append(0)
            for k, angle in enumerate([90, 180, 270]):
                img = self.cifar.__getitem__(i)[0]
                if not clean_data:
                    img = add_arrow(img)
                self.data.append(TF.rotate(img, angle))
                self.labels.append(k + 1)  # Add the rest of labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class chromaCIFAR(Dataset):
    """Make CIFAR with chromatic aberration"""
    def __init__(self, train=True):
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train = train
        if self.train:
            self.cifar = torchvision.datasets.CIFAR10(root = path/'data', download = True, transform=transform, train = True)
        else:
            self.cifar = torchvision.datasets.CIFAR10(root = path/'data', download = True, transform=transform, train = False)
        self.data = []
        self.labels = []
        move = 1
        for i in tqdm(range(len(self.cifar))):
            img, orig_label = self.cifar.__getitem__(i)
            img_zeros = torch.zeros(3,32,32)
            for k, angle in enumerate([0, 90, 180, 270]):
                img = TF.rotate(img, angle) #Rotate first
                img_zeros[2,:,:] = img[2,:,:] #No touch Red and Blue channel
                img_zeros[0,:,:] = img[0,:,:]
                if angle == 0:
                    img_zeros[1,:,move:32] = img[1,:,0:32-move]
                elif angle == 90:
                    img_zeros[1,0:32-move,:] = img[1,1:32,:]
                elif angle == 180:
                    img_zeros[1,:,0:32-move] = img[1,:,1:32]
                elif angle == 270:
                    img_zeros[1,move:32,:] = img[1,0:32-move,:]
                self.data.append(img_zeros)
                self.labels.append(k) #add label
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]