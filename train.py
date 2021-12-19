import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from src.utils.data_loader import VitalDataset
from src.utils.train_function import train

batch_size = 32
input_size = 7
hidden_units = 256
layers = 1
epochs = 1

f = h5py.File('./data/dataset/test_2021-12-17-02:35.npz', 'r')

data_x = f['train']['x']
data_y = f['train']['y']
print("Data x shape: {}".format(data_x.shape))
print("Data y shape: {}".format(data_y.shape))

vital_dataset = VitalDataset(x_tensor=data_x, y_tensor=data_y)
train_loader = DataLoader(dataset=vital_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

model = ValinaLSTM().to(device)
# model.double()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

train(train_loader=train_loader, model=model, epochs=epochs, optimizer=optimizer, loss_fn=criterion, summary_path='./logs/test1/', step_counter=1000)