import torch
import torch.nn as nn
from net import Net
from pytorch.ppg_feature_dataset import FeatureDataset

feature_set = FeatureDataset('newHi-Modnew.csv')

train_loader = torch.utils.data.DataLoader(feature_set, batch_size=10, shuffle=True)

model = Net()
model.to(cur_device)

loss_func = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for feature, labels in train_loader:

        output = model(features)
        loss = loss_func(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


