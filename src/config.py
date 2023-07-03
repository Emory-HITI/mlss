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

TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 128
NUM_WORKER = 30

if torch.cuda.is_available():
    DEVICE = 'cuda:1'
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

DENSENET201 = torchvision.models.densenet201(weights=torchvision.models.DenseNet201_Weights.DEFAULT)
DENSENET201.classifier = nn.Sequential(
        nn.Linear(in_features=(DENSENET201.classifier.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )

EFFICIENTNETB0 = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
EFFICIENTNETB0.classifier = nn.Sequential(
        nn.Linear(in_features=(EFFICIENTNETB0.classifier[1].in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )


RESNET18 = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
RESNET18.fc = nn.Sequential(
        nn.Linear(in_features=(RESNET18.fc.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )

RESNET34 = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.DEFAULT)
RESNET34.fc = nn.Sequential(
        nn.Linear(in_features=(RESNET34.fc.in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )

RESNET101 = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
RESNET101.fc = nn.Sequential(
        nn.Linear(in_features=(RESNET101.fc.in_features),out_features=512),
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


ALEXNET = models.alexnet(pretrained=True)
ALEXNET.classifier = nn.Sequential(
        nn.Linear(in_features=(ALEXNET.classifier[0].in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )


VGG16 = models.vgg16(pretrained=True)
VGG16.classifier = nn.Sequential(
        nn.Linear(in_features=(VGG16.classifier[0].in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )


SQUEEZENET = models.squeezenet1_0(pretrained=True)
SQUEEZENET.classifier = nn.Sequential(
        nn.Linear(in_features=(SQUEEZENET.classifier[0].in_features),out_features=512),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=512,out_features=256),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=256,out_features=128),
        nn.ReLU(inplace=True),
        nn.Linear(in_features=128,out_features=NUM_CLASSES)
    )