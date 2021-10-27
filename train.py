from time import sleep

import numpy as np
import torch
from tqdm import tqdm


def evaluate_model(configs, test_loss, test_correct, test_total, epoch):
    configs.model.eval()
    test_loader = configs.data_loader['valid']
    print("\n------- VALIDATION -------\n")
    with configs.experiment.test():
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

                    tepoch.set_postfix(
                        test_loss=test_loss, test_accuracy=100. * (test_correct/test_total))
        test_loss = round(
            test_loss/len(configs.data_loader['valid'].dataset), 4)
        test_acc = round(((test_correct/test_total) * 100), 4)

        configs.experiment.log_metric("accuracy", test_acc)
        configs.experiment.log_metric("loss", test_loss)

    return [test_loss, test_acc]


def training(configs):
    """
    Performs training and evaluation.
    """

    # setup tracking arrays
    train_acc_arr, test_acc_arr = [], []
    train_loss_arr, test_loss_arr = [], []
    best_test_acc = 0.0
    batch_label_acc = 't_batch_accuracy' if configs.initialization == 1 else 'p_batch_accuracy'
    batch_label_loss = 't_batch_loss' if configs.initialization == 1 else 'p_batch_loss'
    epoch_label_acc = 't_accuracy' if configs.initialization == 1 else 'p_accuracy'
    epoch_label_loss = 't_loss' if configs.initialization == 1 else 'p_loss'

    with configs.experiment.train():

        print("Logging weights as histogram (before training)...")
        # Log model weights
        weights = []
        for name in configs.model.named_parameters():
            if 'weight' in name[0]:
                weights.extend(name[1].detach().numpy().tolist())
        configs.experiment.log_histogram_3d(weights, step=0)

        step = 0
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

                    tepoch.set_postfix(train_loss=loss.item(),
                                       train_accuracy=100. * accuracy)
                    sleep(0.1)

                    # Log batch_accuracy to Comet.ml; step is each batch
                    step += 1

                    configs.experiment.log_metric(
                        batch_label_acc, accuracy, step=step)
                    configs.experiment.log_metric(
                        batch_label_loss, loss.item(), step=step)

                # compute train loss and accuracy
                train_loss = round(
                    train_loss/len(configs.data_loader['train'].dataset), 4)
                train_acc = round(((train_correct/train_total) * 100.0), 4)

                # Log train_loss and train_accuracy to Comet.ml; step is each epoch
                configs.experiment.log_metric(
                    epoch_label_loss, train_loss, step=epoch)

                configs.experiment.log_metric(
                    epoch_label_acc, train_acc, step=epoch)

            print("Logging weights as histogram...")
            # Log model weights
            weights = []
            for name in configs.model.named_parameters():
                if 'weight' in name[0]:
                    weights.extend(name[1].detach().numpy().tolist())
            configs.experiment.log_histogram_3d(weights, step=epoch + 1)

            # evaluate model
            if(configs.initialization == 1):
                test_loss, test_acc = evaluate_model(
                    configs, test_loss, test_correct, test_total, epoch)

                test_acc_arr.append(test_acc)
                test_loss_arr.append(test_loss)

            if(configs.initialization == 1):
                if(test_acc > best_test_acc):
                    print(f"Saving Transfer Model at Epoch: {epoch}")
                    best_test_acc = test_acc
                    torch.save(configs.model.module.state_dict(),
                               configs.save_path)
            else:
                torch.save(configs.model.module.state_dict(),
                           configs.save_path)

            # update tracking arrays
            train_acc_arr.append(train_acc)
            train_loss_arr.append(train_loss)

            if(configs.initialization == 1 and float(test_acc) >= configs.target_val_accuracy):
                break

    return np.asarray(train_acc_arr), np.asarray(train_loss_arr)
