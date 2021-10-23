from time import sleep

import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(configs, test_loss, test_correct, test_total, epoch):
    configs.model.eval()
    test_loader = configs.data_loader['valid']
    with tqdm(test_loader, unit="batch") as tepoch:
        with torch.no_grad():
            for data, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                data, labels = data.cuda(), labels.cuda()

                output = configs.model(data)
                loss = configs.criterion(output, labels)

                test_loss += loss.item()*data.size(0)

                # get the predictions for each image in the batch
                preds = torch.max(output, 1)[1]
                # get the number of correct predictions in the batch
                test_correct += np.sum(np.squeeze(
                    preds.eq(labels.data.view_as(preds))).cpu().numpy())

                # accumulate total number of examples
                test_total += data.size(0)
    test_loss = round(
        test_loss/len(configs.data_loader['test'].dataset), 4)
    test_acc = round(((test_correct/test_total) * 100), 4)

    return [test_loss, test_acc]


def training(configs):
    """
    Performs training and evaluation.
    """

    # setup tracking arrays
    train_acc_arr, test_acc_arr = [], []
    train_loss_arr, test_loss_arr = [], []

    for epoch in range(1, configs.num_epochs+1):

        # tracking variables
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0
        test_loss, test_correct, test_total = 0.0, 0.0, 0.0

        # train the model
        configs.model.train()
        train_loader = configs.data_loader['train']
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, labels in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                # move the data and labels to gpu
                data, labels = data.cuda(), labels.cuda()

                configs.optimizer.zero_grad()
                # get model outputs
                output = configs.model(data)
                # calculate the loss
                loss = configs.criterion(output, labels)
                # backprop
                loss.backward()
                # optimize the weights
                configs.optimizer.step()
                # update the training loss for the batch
                train_loss += loss.item()*data.size(0)
                # get the predictions for each image in the batch
                preds = torch.max(output, 1)[1]
                # get the number of correct predictions in the batch
                curr_correct = np.sum(np.squeeze(
                    preds.eq(labels.data.view_as(preds))).cpu().numpy())

                train_correct += curr_correct
                # accumulate total number of examples

                train_total += data.size(0) if configs.initialization == 1 else (
                    data.size(0)*data.size(1))

                accuracy = train_correct/train_total

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)

            # compute train loss and accuracy
            train_loss = round(
                train_loss/len(configs.data_loader['train'].dataset), 4)
            train_acc = round(((train_correct/train_total) * 100.0), 4)

        # evaluate model
        if(configs.initialization == 1):
            test_loss, test_acc = evaluate_model(
                configs, test_loss, test_correct, test_total, epoch)

            test_acc_arr.append(test_acc)
            test_loss_arr.append(test_loss)

        torch.save(configs.model.module.state_dict(),
                   configs.save_path)

        # update tracking arrays
        train_acc_arr.append(train_acc)
        train_loss_arr.append(train_loss)

        if(configs.initialization == 1 and float(test_acc) >= configs.target_val_acc):
            # print(
            #     f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Acc: {train_acc}% \tTest Loss: {test_loss} \tTest Acc: {test_acc}%")
            break

    return np.asarray(train_acc_arr), np.asarray(train_loss_arr)
