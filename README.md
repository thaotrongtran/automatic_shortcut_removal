# PyTorch Implementation: Automatic shortcut removal for self-supervised representation learning

The experiment on CIFAR-10 is part of the paper, [Automatic shortcut removal for self-supervised
representation learning](https://arxiv.org/abs/2002.08822)


<img src="https://raw.githubusercontent.com/thaotrongtran/automatic_shortcut_removal/master/visualization/lens1.png" width="200" height="200">
<img src="https://raw.githubusercontent.com/thaotrongtran/automatic_shortcut_removal/master/visualization/lens2.png" width="200" height="200">


## Installation
Required python packages are in `requirements.txt`. Install using conda:
```
conda install --file requirements.txt
```

## Training
Possible commandline arguments are defined and described in `arguments.py`.
The default values are:
```
downstream: False (whether training downstream task or not)
clean_data: False (whether training on clean data or not)
shortcut: "arrow" (choose between "arrow" and "chromatic")
lens_usage: False (whether training with Lens network or not)
full_adversarial: False (whether training with full adversarial loss or not)
lambda_term: 1e-10
lr: 0.01
epochs: 50
batch_size: 256
output_dir: 'checkpoints'
```
To train the pretext task, modify dataset and architecture before running if needed. For example, to train lens network, with `--model_name` of '001' to name output model file in the output folder.
```
python train.py --lens_usage --model_name 001
```

To train downstream task of image classification on CIFAR-10 train dataset using weights from model with `--model_name` of '001':
```
python train.py --downstream --model_name 001
```


## Evaluation

Possible commandline arguments are defined and described in `arguments.py`.
The default values are:
```
downstream: False (whether evaluating downstream task or not)
lens_usage: False (whether evaluating with Lens network or not)
batch_size: 256
output_dir: 'checkpoints'
```
A trained model with `--model_name` of '001' can be evaluated on pretext task of rotation:
```
 python eval.py --lens_usage --model_name 001
```

A trained downstream model can be evaluated on the image classification task of CIFAR-10 test dataset:

```
 python eval.py --downstream --model_name 001
```

## Visualization
Use the notebook in folder `Visualization` to see the difference between original image and the Lens network's output.



## References
```
@inproceedings{Minderer2020AutomaticSR,
  title={Automatic Shortcut Removal for Self-Supervised Representation Learning},
  author={Matthias Minderer and Olivier Bachem and Neil Houlsby and Michael Tschannen},
  booktitle={ICML},
  year={2020}
}
```