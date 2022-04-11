import torch
from datasets.cifar import arrowedCIFAR
from models.extractor import Resnet_FC
from models.lens import Unet_ResNet
from arguments import parse_args
from pathlib import Path
import torch.nn as nn
path = Path('.')

def eval_loop(lens_usage, model2, testloader, device, model1 = None):
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
    testset = arrowedCIFAR(train=False, clean_data=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)
    output_dir = Path(path / f'{args.output_dir}')
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

#TODO: Evaluate downstream
def eval_downstream(args):
    pass

if __name__ == '__main__':
    args = parse_args(mode='eval')
    print('Evaluating with these arguments', args)
    if args.downstream:
        eval_downstream(args)
    else:
        evaluate_pretext(args)