import os

import torch
import torch.nn as nn
import torchvision.models as models

from resnet import ResNet50

if __name__ == '__main__':
    model = ResNet50(num_classes=4, transfer=False)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load(
        './model_weights/ag-5-freeze-r50-c10.pt', map_location=torch.device('cpu')))

    model1 = ResNet50(num_classes=10, transfer=False)
    model1.fc = nn.Linear(model1.fc.in_features, 10)
    model1.load_state_dict(torch.load(
        './model_weights/fine-ag-5-freeze-r50-c10.pt', map_location=torch.device('cpu')))

    freeze_count = 5
    count = 0

    for child in model.children():
        count += 1
        print(f"---------------- Count: {count} ----------------")
        print(child)
    # for child, child1 in zip(model.children(), model1.children()):
    #     count += 1
    #     if count >= freeze_count:
    #         print(child)
    #         for param, param1 in zip(child.parameters(), child1.parameters()):
    #             # print(param == param1)
    #             continue
