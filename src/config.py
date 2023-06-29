import torch
import torchvision
import torch.nn as nn

EXP_NAME = "V0" # Version 0 (Change this to whatever you want)

TRAIN_PATH = '../data/chest_xray/train/'
VALID_PATH = '../data/chest_xray/val/'
TEST_PATH = '../data/chest_xray/test/'

LOG_DIR = f'../logs/{EXP_NAME}'
MODEL_DIR = f"../models/{EXP_NAME}"

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
NUM_WORKER = 8

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


EPOCHS = 100

SAVE_WEIGHTS_INTERVAL = 20

LR = 1e-4

LABEL_ENCODING = {
    'NORMAL': 0,
    'PNEUMONIA': 1
}

NUM_CLASSES = 2

LOSS_WEIGHTS = torch.tensor([1.9448173 , 0.67303226],dtype=torch.float).to(DEVICE) # Calculated on the training 



DENSENET121 = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
DENSENET121.classifier = nn.Sequential(
        nn.Linear(in_features=(DENSENET121.classifier.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )


RESNET50 = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
RESNET50.fc = nn.Sequential(
        nn.Linear(in_features=(RESNET50.fc.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )