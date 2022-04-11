from pathlib import Path
from models.lens import Unet_ResNet
from models.extractor import Resnet_FC
from datasets.cifar import arrowedCIFAR
import torch
# import matplotlib.pyplot as plt
# import numpy as np
from tqdm import tqdm
# from torchsummary import summary

import torch.nn as nn
# from torch import Tensor
# from torchvision.models.resnet import BasicBlock,Bottleneck
import torch.optim as optim

path = Path('.')


def recon_loss(raw_inputs, lens_output):
    loss = nn.MSELoss(reduction = 'sum')
    return loss(raw_inputs,lens_output)

def lens_loss(raw_inputs, lens_output, lambda_term, ssl_loss = None, min_probs = None, final_outputs = None):
    #Adversarial loss: two types
    if ssl_loss:
        total_loss = -ssl_loss + lambda_term*recon_loss(raw_inputs, lens_output)
    else:
        celoss = nn.CrossEntropyLoss(reduction='mean')
        adv_loss = celoss(final_outputs,min_probs)
        total_loss = adv_loss + lambda_term*recon_loss(raw_inputs, lens_output)
    return total_loss


def train_loop(trainloader, device, lens_usage, model2, num_epochs,learning_rate, lambda_term, model1 = None, full_adversarial = False):
    if lens_usage:
        optim1 = optim.Adam(model1.parameters(), lr=learning_rate, betas=(0.1, 0.001), eps=1e-07)
        model1.train()
    optim2 = optim.Adam(model2.parameters(), lr=learning_rate, betas=(0.1, 0.001), eps=1e-07)
    model2.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    sm = nn.Softmax(dim=1)
    for epoch in range(num_epochs):
        ssl_losses = 0.0
        lens_losses = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # Zero gradients out
            if lens_usage:
                optim1.zero_grad()
            optim2.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            if lens_usage:
                lens_output = model1(inputs)
                lens_out_detach = lens_output.detach()
                lens_out_detach.requires_grad_(True)
                outputs = model2(lens_out_detach)
                # For type 2 of Adversarial loss
                min_probs = torch.argmin(sm(outputs), dim=1)
                ssl_loss = criterion(outputs, labels)
                if full_adversarial:
                    l_loss = lens_loss(inputs, lens_out_detach, lambda_term = lambda_term, ssl_loss = ssl_loss)
                else:
                    l_loss = lens_loss(inputs, lens_out_detach, lambda_term=lambda_term, min_probs=min_probs,
                                   final_outputs=outputs)
                l_loss.backward(retain_graph=True)
                lens_output.backward(lens_out_detach.grad)  # Let the grad of l_loss go thru
                optim2.zero_grad()  # Clear out l_loss grad from model2
                ssl_loss.backward()
            else:
                outputs = model2(inputs)
                ssl_loss = criterion(outputs, labels)
                ssl_loss.backward()
            # Update step
            if lens_usage:
                optim1.step()
                lens_losses += l_loss.item()
            optim2.step()
            ssl_losses += ssl_loss.item()
            if i > 0 and i % 50 == 0:
                print(f'Epoch: {epoch}, batch {i}] ssl_loss: {ssl_losses / i:.3f} lens_loss: {lens_losses / i:.3f}')

def train_model(clean_data, num_epochs, batch_size, learning_rate, lambda_term, lens_usage, model_name, full_adversarial= False):
    trainset = arrowedCIFAR(train=True, clean_data=True)
    batch_size = 512
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)

    if lens_usage:
        model1 = Unet_ResNet()
        model1.to(device)
    else:
        model1 = None
    model2 = Resnet_FC(out_classes=4)
    model2.to(device)
    train_loop(trainloader, device, lens_usage, model2, num_epochs, learning_rate, lambda_term, model1=model1,
               full_adversarial=full_adversarial)
    #Saving model
    if lens_usage:
        torch.save(model1, path/f'lens_{model_name}.pth')
    torch.save(model2, path / f'extractor_{model_name}.pth')

if __name__ == '__main__':
    #TODO: arg for getting arrowed CIFAR
    #Clean data or not


