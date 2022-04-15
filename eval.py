import torch
from datasets.cifar import arrowedCIFAR
from models.extractor import Resnet_FC
from models.lens import Unet_ResNet
from models.logistic import Logistic_Net
from arguments import parse_args
from pathlib import Path
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

path = Path('.')

def eval_loop(lens_usage, model2, testloader, device, model1 = None):
    '''
    Pretext evaluation loop
    '''
    correct = 0
    total = 0
    model2.eval()
    model1.eval()  # SETTING EVAL MODE
    sm = nn.Softmax(dim=1)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if lens_usage:
                outputs = model2(model1(images.to(device)))
            else:
                outputs = model2(images.to(device))
            predicted = torch.argmax(sm(outputs), dim=1).cpu()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy on test images: {100 * correct // total} %')

def evaluate_pretext(args):
    '''
    Method to evaluate a model on pretext task
    '''
    testset = arrowedCIFAR(train=False, clean_data=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)
    output_dir = Path(path / f'{args.output_dir}/{args.model_name}')
    if output_dir.exists():
        if args.lens_usage:
            model1 = torch.load(f'{args.output_dir}/{args.model_name}/lens.pth')
            model1.to(device)
        else:
            model1 = None
        model2 = torch.load(f'{args.output_dir}/{args.model_name}/extractor.pth')
        model2.to(device)
        eval_loop(args.lens_usage, model2, testloader, device, model1 = model1)
    else:
        print('Model output folder does not exist. Please check again.')

def eval_downstream(args):
    '''
    Method to evaluate a model downstream task
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)
    output_dir = Path(path / f'{args.output_dir}/{args.model_name}')
    if output_dir.exists():
        model =  torch.load(path/f'{args.output_dir}/{args.model_name}/logistic.pth')
        model.to(device)
        sm = nn.Softmax(dim=1)
        # Evaluation
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images.to(device))
                predicted = torch.argmax(sm(outputs), dim=1).cpu()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy on test images: {100 * correct // total} %')
    else:
        print('Model output folder does not exist. Please check again.')

if __name__ == '__main__':
    args = parse_args(mode='eval')
    print('Evaluating with these arguments', args)
    if args.downstream:
        eval_downstream(args)
    else:
        evaluate_pretext(args)