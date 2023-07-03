import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import os
from glob import glob
import numpy as np
from tqdm import tqdm

import config
from dataset import CXR_Dataset
from validate import valid
from training import train

import glob
from datetime import datetime

def run(model, model_name, optimizer, epochs=config.EPOCHS):
    
    train_images = glob.glob(config.TRAIN_PATH + "*/*.jpeg")
    valid_images = glob.glob(config.VALID_PATH + "*/*.jpeg")
    
    
    # Use Albumation lib for augmentations
    train_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config.IMAGE_HEIGHT + 30, config.IMAGE_WIDTH + 30)),
                    transforms.RandomCrop(size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomHorizontalFlip(),
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.05, 2)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    valid_transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((config.IMAGE_HEIGHT, config.IMAGE_WIDTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    train_dataset = CXR_Dataset(image_paths=train_images, transforms=train_transform)
    valid_dataset = CXR_Dataset(image_paths=valid_images, transforms=valid_transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKER,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKER,
    )
    
    current_time = datetime.now()
    
    save_dir = os.path.join(config.MODEL_DIR, f'{model_name}/{current_time}')
    writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, f'{model_name}/{current_time}'))
    min_valid_loss = 1e10
    
    for epoch in range(epochs):
        
        training_loss, training_accuracy, training_f1 = train(model, optimizer, train_dataloader)
        validation_loss, validation_accuracy, validation_f1 = valid(model, valid_dataloader)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if epoch % config.SAVE_WEIGHTS_INTERVAL == 0 and epoch > 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth.tar'))

        if validation_loss < min_valid_loss:
            print("*"*70)
            print(
                f"Validation loss descreased from {min_valid_loss} to {validation_loss}"
            )
            
            # Saving a lot more than just the state_dict
            save_config = {
                #"model_arch": model.detach().cpu(),
                "model_name": model_name,
                "model_weights": model.state_dict(),
                "epoch": epoch,
                "optimizer": optimizer,
            }
            torch.save(save_config, os.path.join(save_dir, "best_model_config.pth.tar"))

            min_valid_loss = validation_loss

            print("Saving the model")
            print("*"*70)
            print("*"*70)

        print(
            f"Epoch: {epoch} | Training Loss: {training_loss} | Training Accuracy: {training_accuracy} | Training F1Score: {training_f1} | Validation Loss: {validation_loss} | Validation Accuracy: {validation_accuracy} | Validation F1Score: {validation_f1}"
        )

        writer.add_scalar("Accuracy/Train", training_accuracy, epoch)
        writer.add_scalar("Accuracy/Valid", validation_accuracy, epoch)
        writer.add_scalar("Loss/Train", training_loss, epoch)
        writer.add_scalar("Loss/Valid", validation_loss, epoch)

        for i in range (config.NUM_CLASSES):
            writer.add_scalar(f"F1 score/Train/{i}", training_f1[i].item(), epoch)
            writer.add_scalar(f"F1 score/Valid/{i}", validation_f1[i].item(), epoch)
        
        
if __name__ == "__main__":
    
    model = config.DENSENET121
    model.to(device=config.DEVICE)
    model_name = "densenet121"
    
    print("*"*100)
    print("*"*100)
    print(model_name)
    print("-"*100)
    print(model)
    print("*"*100)
    print("*"*100)
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LR)
    run(model, model_name, optimizer)