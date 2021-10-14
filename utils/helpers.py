import os

import torch
import torch.nn as nn
import torchvision.models as models
from resnet import ResNet50


def config_dict(location):
    """Function reads the configs.txt file and returns
       a dictionary with all the hyperparameters key, value
       pairs. """

    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))

    # file = open(os.path.join(__location__, location))
    file = open(location)
    parameters = file.readlines()

    params = {}
    for parameter in parameters:
        # print(parameter)
        # remove whitespace
        parameter = parameter.replace(" ", "")
        parameter = parameter.replace("\n", "")
        # get the key and value
        key, value = parameter.split("=")

        # convert value to the right type
        value = cast_type(value)

        params[key] = value

    return params


def cast_type(s):
    """ Function casts the string value to the right type
    for the hyperparameters"""

    s = str(s.replace(" ", ""))

    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            s = str(s)

    return s


def load_model(configs, classes):
    """ Function loads the model based on hyperparameters"""

    if configs.initialization:
        print("Loading arch for initialization: {}, Type: {}.".format(
            configs.arch, type(configs.arch)))
        print("Loading weights for arch from: {}".format(
            configs.model_weights_dir))

        # load model
        model, _ = load_models(configs.arch)

        # update fc layer with pretraining classes
        # model.fc = nn.Linear(model.fc.in_features, configs.num_classes)

        # # load weights from pretraining
        # model.load_state_dict(torch.load(
        #     configs.model_weights_dir + configs.model_in_name))

        # # update the fc layer for transfer
        # model.fc = nn.Linear(model.fc.in_features, classes)

        return model, configs.model_out_name
    else:
        print("Loading arch for training: {}, Type: {}.".format(
            configs.arch, type(configs.arch)))
        # load model
        model, _ = load_models(configs.arch)
        # update final layer
        model.fc = nn.Linear(model.fc.in_features, classes)

        return model, configs.model_out_name


def load_models(arch):
    """
    This function returns an architecture based on user input.
    params:
        - arch (str) : string which acts as a keyword for loading and architecture.
    returns:
        - architecture (torch Model) : torch model based on keyword.
        - final_layer (int) : returns the number of neurons in the last fc layer.
    """
    if arch == "alexnet":
        return models.alexnet(), 4096
    elif arch == "vgg19":
        return models.vgg19_bn(), 4096
    elif arch == "inception":
        return models.inception_v3(transform_input=True)
    elif arch == "resnet18":
        return models.resnet18(), 512
    elif arch == "resnet50":
        return models.resnet50(), 512
    elif arch == "resnet152":
        return models.resnet152(), 512
    elif arch == "wide_resnet50_2":
        return models.wide_resnet50_2(), 2048
    elif arch == "resnet50_scratch":
        return ResNet50(num_classes=4), 512
