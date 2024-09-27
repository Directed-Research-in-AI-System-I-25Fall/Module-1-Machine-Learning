import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb
from argparse import ArgumentParser
import pickle

from scripts.prepare_data import prepare_cifar10_datasets
from models.SimpleCNN import SimpleCNN
from models.SimpleViT import SimpleViT
from models.SimpleComposition import SimpleComposition


def count_parameters(model, except_linear=False):
    if except_linear:
        total_params = 0
        for layer in model.children():
            if not isinstance(layer, nn.Linear):
                total_params += sum(p.numel() for p in layer.parameters())
        return total_params
    else:
        return sum(p.numel() for p in model.parameters())


def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=10):
    model.to(device)
    model_save_path = f"./model_ckpt/cifar10_{model.name()}.pth"
    best_acc = -1.
    
    
    for epoch in range(num_epochs):
        # TODO: Implement the training loop. Remember to call schedulers if they exist; and remeber to record your losses and metrics using WandB.
        raise NotImplementedError()

        # TODO: Implement the Validation phase. Remember to save model state_dict into path `model_save_path` at the best acc; and remeber to record your losses and metrics using WandB.
        raise NotImplementedError()






# Main script
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="SimpleCNN", type=str)
    args = parser.parse_args()

    # TODO: Fix all relevant random seeds for reproducibility.
    raise NotImplementedError()

    # TODO: Set your device.
    device = None
    
    # TODO: Initialize your WandB settings.
    raise NotImplementedError()

    # TODO: Set model hyperparameters, and batch_size,
    #       and initialize `model`, `optimizer`, `scheduler`, 
    #       and `train_transform` (used to initialize train_dataset. You can add data augmentation here. 
    #       You can use all functions in torchvision.transforms. If `train_transform` is set to None, it defaults to the same transform used in the test dataset 
    #       as defined in `prepare_cifar10_datasets`.
    if args.model == "SimpleCNN":
        model_hps = {
            "conv_dim1": None,
            "conv_dim2": None,
            "fc_dim": None,
            "dropout_rate": None,
        }
        batch_size = None
        train_transform = None
        
        model = SimpleCNN(**model_hps)
        
        total_params = count_parameters(model)
        print(f'Total number of parameters: {total_params}')
        assert total_params < 100_000, "Please make sure your CNN model's total param num is lower than 100K."
        
        optimizer = None
        scheduler = None
        
    elif args.model == "SimpleViT":
        model_hps = {
            "depth": None,
            "heads": None,
            "mlp_dim": None,
            "dim_head": None,
            "dropout": None,
            "emb_dropout": None,
            "image_size": None,
            "patch_size": None,
            "channels": None,
            "dim": None
        }
        batch_size = None
        train_transform = None
        
        model = SimpleViT(**model_hps)
        
        total_params = count_parameters(model)
        print(f'Total number of parameters: {total_params}')
        
        # TODO: Init optimizer and scheduler (Optionally for scheduler. If you do not use scheduler, set it to None.)
        optimizer = None
        scheduler = None
        
    elif args.model == "SimpleComposition":
        model_hps = {
            "CNN_model_hps": {
                "conv_dim1": None,
                "conv_dim2": None,
                "fc_dim": None,
                "dropout_rate": None,
            },
            "ViT_model_hps": {
                "depth": None,
                "heads": None,
                "mlp_dim": None,
                "dim_head": None,
                "dropout": None,
                "emb_dropout": None,
                "image_size": None,
                "patch_size": None,
                "channels": None,
                "dim": None
            }
        }
        batch_size = None
        train_transform = None
        
        model = SimpleComposition(model_hps["CNN_model_hps"], model_hps["ViT_model_hps"])
        
        total_params = count_parameters(model.cnn, except_linear=True) + count_parameters(model.vit, except_linear=False) 
        print(f'Total number of parameters: {total_params}')
        assert total_params < 120_000, "Please make sure your Composition model's total param num is lower than 120K."
        
        # TODO: Init optimizer and scheduler (Optionally for scheduler. If you do not use scheduler, set it to None.)
        optimizer = None
        scheduler = None
        
    else:
        raise ValueError(f"Model {args.model} is not implemented!")
    
    model_ckpt_save_path = f"./model_ckpt/cifar10_{model.name()}_arg.pickle"
    # TODO: Save the `model_hps` to file `model_ckpt_save_path` for testing. Use pickle.
    raise NotImplementedError()
    
    # Prepare datasets
    train_dataset, valid_dataset, _ = prepare_cifar10_datasets(train_transform)

    # TODO: Initialize corresponding dataloaders. Be careful whether to shuffle in each of the datasets.
    # NOTE: If you're applying more advanced data augmentation in the "transform", consider increasing 'num_workers'
    # to parallelize data processing during training for faster training.
    train_loader = None
    valid_loader = None
    
    
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    import time
    start_time = time.time()
    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=75)
    
    print(f"Training takes {(time.time() - start_time) / 60} min")
    
    # Finish WandB run
    wandb.finish()
