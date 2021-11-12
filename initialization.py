
import random

import numpy as np
import torch
from comet_ml import Experiment
from torch import nn, optim

from loaders import initialization_loaders
from train import training
from utils.config import Config
from utils.helpers import config_dict, load_model


def initialization(configs):
    """
    This function runs the initialization training framework.
    params:
        - configs (Config) : config object used for training.
    """

    p_data_loader, t_data_loader, _, t_classes = initialization_loaders(configs,
                                            configs.dataset,
                                            num_workers=configs.num_workers)
    ######################### Pretraining #########################

    # saved pretraining-dataloader into configs.
    configs.data_loader = p_data_loader

    # Load the loss criterion.
    configs.criterion = nn.CrossEntropyLoss()

    # Update Pretraining Epochs
    configs.num_epochs = configs.p_num_epochs

    # run pretraining.
    pretraining(configs, classes=configs.p_num_classes)

    # update num_classes for pretraining
    configs.num_classes = configs.p_num_classes

    print("Pre-Training complete.")
    print("Initializing the weights for transfer/retraining now!")

    ##################### Transfer/Retraining #####################
    torch.cuda.empty_cache()

    # saved dataloaders into configs.
    configs.data_loader = t_data_loader
    configs.t_classes = t_classes
    print(f"Length t classes: {len(t_classes)}")
    # run transfer learning or retraining.
    transfer_and_retrain(configs, classes=len(t_classes))

    print("Transfer learning complete.")
    print("Model is ready for evaluation.")


def pretraining(configs, classes=0):
    """
    This function runs the pretraining for the model.
    params:
        - configs (Config) : config object used for training.
        - classes (int) : classes for training.
    """
    # load pretraining model.
    pretraining_model, pretraining_model_name = load_model(configs, classes)
    print(f"Model for {classes} pretrain classes loaded successfully!")

    configs.experiment.set_model_graph(str(pretraining_model))

    # move the model to GPU and DataParallel if possible.

    if torch.cuda.device_count() > 1:
        pretraining_model = nn.DataParallel(pretraining_model)
        print("\nPretraining model moved to Data Parallel")
    pretraining_model.cuda()

    # create the optimizer.
    if(configs.adam == 1):
        optimizer = torch.optim.Adam(
            pretraining_model.parameters(),
            lr=configs.p_lr,
            weight_decay=configs.p_weight_decay)
    else:
        optimizer = torch.optim.SGD(
            pretraining_model.parameters(),
            momentum=configs.p_momentum,
            lr=configs.p_lr,
            weight_decay=configs.p_weight_decay
        )

    # save model and optimizer in configs.
    configs.model = pretraining_model
    configs.optimizer = optimizer

    # create scheduler dictionary.
    configs.schedule_dict = {}

    if configs.adam == 0:
        print("Schedule Dict for Pretrain")
        print(configs.schedule_dict)
    # Pretrain the model and save the weights.
    save_path = configs.model_weights_dir + "pretrain/" + pretraining_model_name
    print("Pretraining model will be saved at {}".format(save_path))

    configs.save_path = save_path

    # run training for the pretrain model.
    training(configs)


def transfer_and_retrain(configs, classes=0):
    """
    This function performs trasnfer learning/ retraining for pretrained model.
    params:
        - configs (Config) : config object used for training.
        - classes (int) : number of classes used for transfer/retrain.
    """
    # set the training to transfer mode.
    configs.initialization = 1
    # reset number of epochs.
    configs.num_epochs = configs.t_num_epochs
    # reset the target validation accuracy
    configs.target_val_accuracy = 95.0
    # load transfer model.
    transfer_model, transfer_model_name = load_model(configs, classes)
    configs.experiment.set_model_graph(str(transfer_model))
    # update num_classes to transfer classes
    configs.num_classes = classes

    print(
        f"\nTransfer model for {configs.num_classes} classes loaded")

    # move the model to GPU and DataParallel if possible.
    if configs.gpu_avail:
        if torch.cuda.device_count() > 1:
            transfer_model = nn.DataParallel(transfer_model)
            print("\nTransfer model moved to Data Parallel")
            transfer_model.cuda()
        else:
            raise ValueError("Train on GPU is recommended!")

    for child in transfer_model.children():
        for param in child.parameters():
            if (param.requires_grad == True):
                print(child)

    # create the optimizer.
    if configs.adam:
        optimizer = torch.optim.Adam(
            transfer_model.parameters(),
            lr=configs.t_lr,
            weight_decay=configs.t_weight_decay)
    else:
        optimizer = torch.optim.SGD(
            transfer_model.parameters(),
            momentum=configs.t_momentum,
            lr=configs.t_lr,
            weight_decay=configs.t_weight_decay
        )

    # save model and optimizer on configs.
    configs.model = transfer_model
    configs.optimizer = optimizer
    
    configs.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(configs.optimizer, T_max=200)
    

    # train the model and save the weights.
    save_path = configs.model_weights_dir + "finetune/" + transfer_model_name
    print(f"Transfer model will be saved at {save_path}")

    configs.save_path = save_path
    # train the transfer/retrain model.
    model = training(configs)

    return model


if __name__ == "__main__":
    params_dict = config_dict('./config.txt')
    configs = Config(params_dict)

    # start comet ML experiment
    experiment = Experiment(
        api_key="ZgD8zJEiZErhwIzPMfZpitMjq",
        project_name="ssl-pretraining",
        workspace="ayushm-agrawal",
    )

    experiment.set_name(configs.experiment_name)
    # experiment.add_tag("initialization")

    # log hyperparameters in comet ML
    experiment.log_parameters(configs)

    # check if dataset is provided.
    if configs.dataset is None:
        raise ValueError(
            "Dataset location not provided")

    # check for GPU.
    configs.gpu_avail = torch.cuda.is_available()

    # set seed for reproducibility.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # # pass the experiment object to configs
    configs.experiment = experiment

    if (configs.initialization == 1):
        _, t_data_loader, _, t_classes = initialization_loaders(configs,
                                            configs.dataset,
                                            num_workers=configs.num_workers)
        configs.data_loader = t_data_loader
        configs.t_classes = t_classes
        print(f"Length t classes: {len(t_classes)}")
        # Load the loss criterion.
        configs.criterion = nn.CrossEntropyLoss()
        # run transfer learning or retraining.
        _ = transfer_and_retrain(configs, classes=len(t_classes))
        
    else:
        # run model training.
        initialization(configs)
    
    

    pretrain_model_log = configs.model_weights_dir + \
        "pretrain/" + configs.model_in_name
    transfer_model_log = configs.model_weights_dir + \
        "finetune/" + configs.model_in_name

    experiment.log_model("Pretrain Model", pretrain_model_log)
    experiment.log_model("Transfer Model", transfer_model_log)
