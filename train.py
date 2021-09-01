import numpy as np
import torch


def training(configs):
    """
    Performs training and evaluation.
    """

    min_test_loss = np.Inf

    # setup tracking arrays
    train_acc_arr, test_acc_arr = [], []
    train_loss_arr, test_loss_arr = [], []

    for epoch in range(1, configs.epochs+1):

        # tracking variables
        train_loss, train_correct, train_total = 0.0, 0.0, 0.0
        test_loss, test_correct, test_total = 0.0, 0.0, 0.0

        # train the model
        configs.model.train()

        for data, labels in configs.loaders['train']:
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
            train_correct += np.sum(np.squeeze(
                preds.eq(labels.data.view_as(preds))).cpu().numpy())

            # accumulate total number of examples
            train_total += data.size(0)

        # compute train loss and accuracy
        train_loss = round(train_loss/len(configs.loaders['train'].dataset), 4)
        train_acc = round(((train_correct/train_total) * 100.0), 4)

        configs.model.eval()
        # with torch.no_grad():
        #     for data, labels in configs.loaders['test']:

        #         data, labels = data.cuda(), labels.cuda()

        #         output = configs.model(data)
        #         loss = configs.criterion(output, labels)

        #         test_loss += loss.item()*data.size(0)

        #         # get the predictions for each image in the batch
        #         preds = torch.max(output, 1)[1]
        #         # get the number of correct predictions in the batch
        #         test_correct += np.sum(np.squeeze(
        #             preds.eq(labels.data.view_as(preds))).cpu().numpy())

        #         # accumulate total number of examples
        #         test_total += data.size(0)

        # # save model
        # if test_loss < min_test_loss:
        #     print(f"Saving model at Epoch: {epoch}")
        torch.save(configs.model.module.state_dict(),
                   configs.save_path+configs.exp_name)

        # compute test loss and accuracy
        # test_loss = round(test_loss/len(configs.loaders['test'].dataset), 4)
        # test_acc = round(((test_correct/test_total) * 100), 4)

        # update tracking arrays
        train_acc_arr.append(train_acc)
        # test_acc_arr.append(test_acc)
        train_loss_arr.append(train_loss)
        # test_loss_arr.append(test_loss)

        # print(
        #     f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Acc: {train_acc}% \tTest Loss: {test_loss} \tTest Acc: {test_acc}%")
        # if float(test_acc) >= configs.target_val_acc:
        #     break
        print(
            f"Epoch: {epoch} \tTrain Loss: {train_loss} \tTrain Acc: {train_acc}% ")

    return np.asarray(train_acc_arr), np.asarray(train_loss_arr)
