import torch
import torch.nn as nn
import config
from tqdm import tqdm
from sklearn.metrics import f1_score
import torchmetrics

def train(
    model,
    optimizer,
    dataloader,
):
    
    training_loss = 0.0
    num_correct = 0
    num_example = 0
    f1_sc = 0
    f1_score = torchmetrics.classification.MulticlassF1Score(num_classes=config.NUM_CLASSES, average=None).to(config.DEVICE)

    model.train()
    for batch in tqdm(dataloader, desc="Training"):
        image = batch["img"].to(device=config.DEVICE)
        targets = batch["target"].to(device=config.DEVICE)

        optimizer.zero_grad()
        outputs = model(image)

        loss = nn.CrossEntropyLoss(weight=config.LOSS_WEIGHTS)(outputs, targets)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.data.item()
        num_correct += (
            (
                torch.argmax(outputs, dim=1) == targets
            ).sum().item())
        
        num_example += targets.size(0)
        f1_sc += f1_score(outputs, targets)
        

    training_loss /= len(dataloader.dataset)
    accuracy = num_correct / num_example
    f1_sc /= len(dataloader.dataset)

    return training_loss, accuracy, f1_sc