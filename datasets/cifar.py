from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm

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

#TODO: chroma dataset