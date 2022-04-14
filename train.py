from pathlib import Path
from models.lens import Unet_ResNet
from models.extractor import Resnet_FC
from models.logistic import Logistic_Net
from datasets.cifar import arrowedCIFAR, chromaCIFAR
import torch
from arguments import parse_args
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

path = Path('.')
def recon_loss(raw_inputs, lens_output):
    '''
    Calculate reconstruction loss for the lens loss
    :param raw_inputs: original unaltered images
    :param lens_output: outputs of the images after lens network
    :return: loss
    '''
    loss = nn.MSELoss(reduction = 'mean')
    return loss(raw_inputs,lens_output)

def adv_loss(ssl_loss = None, min_probs = None, final_outputs = None):
    '''
    Two types of adversarial loss to optmize lens network
    :param ssl_loss: cross entropy loss
    :param min_probs: bias towards least likely class
    :param final_outputs: output of the last layer of the extractor
    :return: adversarial loss
    '''
    if ssl_loss:
        total_loss = -ssl_loss
    else:
        celoss = nn.CrossEntropyLoss(reduction='mean')
        adv_loss = celoss(final_outputs,min_probs)
        total_loss = adv_loss
    return total_loss

def train_loop(trainloader, device, lens_usage, model2, num_epochs,learning_rate, lambda_term, model1 = None,
               full_adversarial = False):
    '''
    Train loop to train based on input config
    '''
    if lens_usage:
        optim1 = optim.Adam(model1.parameters(), lr=learning_rate, betas=(0.1, 0.001), eps=1e-07)
        model1.train()
    optim2 = optim.Adam(model2.parameters(), lr=learning_rate, betas=(0.1, 0.001), eps=1e-07)
    model2.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')
    sm = nn.Softmax(dim=1)
    for epoch in tqdm(range(num_epochs)):
        ssl_losses = 0.0
        adv_losses = 0.0
        recon_losses = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            # Zero gradients out
            if lens_usage:
                optim1.zero_grad()
            optim2.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            if lens_usage:
                #Running through 2 networks
                lens_output = model1(inputs)
                lens_out_detach = lens_output.detach()
                lens_out_detach.requires_grad_(True)
                outputs = model2(lens_out_detach)

                #Prepare for loss function and bacprop
                min_probs = torch.argmin(sm(outputs), dim=1)
                ssl_loss = criterion(outputs, labels)
                if full_adversarial:
                    adv = adv_loss(ssl_loss=ssl_loss)
                else:
                    adv = adv_loss(min_probs=min_probs, final_outputs=outputs)
                adv.backward(retain_graph=True)
                r_loss = lambda_term*recon_loss(inputs, lens_out_detach)
                r_loss.backward()
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
                adv_losses += adv_loss.item()
                recon_losses += r_loss.item()
            optim2.step()
            ssl_losses += ssl_loss.item()
            if i > 0 and i % 50 == 0:
                print(f'[{epoch}, batch {i}] ssl_loss: {ssl_losses / i:.3f} lens_loss: {adv_losses / i:.3f},'
                      f' recon_loss: {recon_losses / i:.3f}')

def train_lens(args):
    '''
    Training network on pretext task with or without lens
    '''
    if args.clean_data:
        trainset = arrowedCIFAR(train=True, clean_data=True)
    elif args.shortcut == 'arrow':
        trainset = arrowedCIFAR(train=True, clean_data=False)
    elif args.shortcut == 'chromatic':
        trainset = arrowedCIFAR(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)
    if args.lens_usage:
        model1 = Unet_ResNet()
        model1.to(device)
    else:
        model1 = None
    model2 = Resnet_FC(out_classes=4)
    model2.to(device)
    train_loop(trainloader, device, args.lens_usage, model2, args.epochs, args.lr, args.lambda_term, model1=model1,
               full_adversarial=args.full_adversarial)
    #Saving model
    output_dir = Path(path/f'{args.output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.lens_usage:
        torch.save(model1, path/f'{args.output_dir}/{args.model_name}/lens.pth')
    torch.save(model2, path / f'{args.output_dir}/{args.model_name}/extractor.pth')

def train_downstream(args):
    '''
    Training downstream classification task
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)
    output_dir = Path(path / f'{args.output_dir}')
    if output_dir.exists():
        if args.lens_usage:
            model = Logistic_Net(path / f'{args.output_dir}/{args.model_name}/extractor.pth',
                                 pretrained_lens_path=path / f'{args.output_dir}/{args.model_name}/lens.pth')
        else:
            model = Logistic_Net(path / f'{args.output_dir}/{args.model_name}/extractor.pth')
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        losses = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            opt.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            losses += loss.item()
            if i > 0 and i % 10 == 0:
                print(f'[{epoch}, batch {i}] loss: {losses / i:.3f}')
    #Saving model
    output_dir = Path(path/f'{args.output_dir}/{args.model_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, path / f'{args.output_dir}/{args.model_name}/logistic.pth')

if __name__ == '__main__':
    args = parse_args(mode='train')
    print('Training with these arguments', args)
    if args.downstream:
        train_downstream(args)
    else:
        train_lens(args)