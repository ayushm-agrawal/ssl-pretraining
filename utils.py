import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import resnet

from resnet import ResNet50


def load_model(configs):
    """ Function loads the model based on hyperparameters"""

    if configs.model_name == "Resnet50":
        print("Loading Resnet18")

        # load model
        # model = models.resnet50()
        model = ResNet50(10)

        # update fc layer with pretraining classes
        # model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

        if configs.finetune:
            print("Loading pretrained weights")
            # load weights from pretraining
            model.load_state_dict(torch.load(configs.pretrain_weights))

            # update the fc layer for transfer
            model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

        # push model to cuda
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print("\nModel moved to Data Parallel")
        model.cuda()

        return model
