from time import sleep

import numpy as np
import torch
from tqdm import tqdm


def training(configs, transfer=False):
    """
    Performs training and evaluation.
    """

    min_test_loss = np.Inf

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
                # print(f"Tensor Bitch: {output}, \n {labels}")
               #  print(f"Shape Bitch: \n{output.shape}, {labels.shape}")
                
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
                if(configs.initialization == 1):
                    train_total += data.size(0)
                else:
                    train_total += (data.size(0)*data.size(1))

                accuracy = train_correct/train_total

                tepoch.set_postfix(loss=loss.item(), accuracy=100. * accuracy)
                sleep(0.1)

            # compute train loss and accuracy
            train_loss = round(
                train_loss/len(configs.data_loader['train'].dataset), 4)
            train_acc = round(((train_correct/train_total) * 100.0), 4)

            if(configs.initialization == 1):
                configs.model.eval()
                with torch.no_grad():
                    for data, labels in configs.data_loader['test']:

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

                # save model
                if test_loss < min_test_loss:
                    print(f"Saving model at Epoch: {epoch}")

            if(configs.arch == 'resnet50_scratch'):
                torch.save(configs.model.module.state_dict(),
                           configs.save_path)
            else:

                torch.save(configs.model.module.state_dict(),
                           configs.save_path)

            # compute test loss and accuracy
            if(configs.initialization == 1):
                test_loss = round(
                    test_loss/len(configs.data_loader['test'].dataset), 4)
                test_acc = round(((test_correct/test_total) * 100), 4)
                test_acc_arr.append(test_acc)
                test_loss_arr.append(test_loss)

            # update tracking arrays
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)

            if(configs.initialization == 1):
                print(
                    f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Acc: {train_acc}% \tTest Loss: {test_loss} \tTest Acc: {test_acc}%")
                if float(test_acc) >= configs.target_val_acc:
                    break

    return np.asarray(train_acc_arr), np.asarray(train_loss_arr)
