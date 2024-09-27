import sys
sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from scripts.prepare_data import prepare_cifar10_datasets
from models.SimpleCNN import SimpleCNN
from models.SimpleViT import SimpleViT
from models.SimpleComposition import SimpleComposition
import pickle



def test_model(model, test_loader, device):
    # TODO: Implement your test procedure.
    raise NotImplementedError()
    
    return accuracy

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="SimpleCNN", type=str)
    args = parser.parse_args()

    # TODO: Set your device.
    device = None
    
    
    # Prepare datasets
    _, _, test_dataset = prepare_cifar10_datasets(None)

    # TODO: Prepare data loader for the test dataset
    test_loader = None



    model_ckpt_save_path = f"./model_ckpt/cifar10_{args.model}_arg.pickle"
    # TODO: Load `model_hps`` with pickle from `model_ckpt_save_path`.
    model_hps = None
    
    if args.model == "SimpleCNN":
        model = SimpleCNN(**model_hps)
    elif args.model == "SimpleViT":
        model = SimpleViT(**model_hps)
    elif args.model == "SimpleComposition":
        model = SimpleComposition(model_hps["CNN_model_hps"], model_hps["ViT_model_hps"])
    else:
        raise ValueError(f"Model {args.model} is not implemented!")
    
    model_save_path = f"./model_ckpt/cifar10_{model.name()}.pth"
    # TODO: Load model state_dicts from `model_save_path`.
    raise NotImplementedError()

    # Test the model
    test_accuracy = test_model(model, test_loader, device)

    print(f'Test Accuracy of the model on the CIFAR-10 test dataset in %: {test_accuracy:.2f}')
