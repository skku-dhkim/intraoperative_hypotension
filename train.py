import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from src.utils.data_loader import VitalDataset
from src.utils.train_function import train
from src.models.rnn import ValinaLSTM

batch_size = 32
input_size = 7
hidden_units = 256
layers = 1
epochs = 1

f = h5py.File('./data/dataset/train_2021-12-17-17:43.hdf5', 'r')

data_x = f['train']['x']
data_y = f['train']['y']
print("Data x shape: {}".format(data_x.shape))
print("Data y shape: {}".format(data_y.shape))

vital_dataset = VitalDataset(x_tensor=data_x, y_tensor=data_y)
train_loader = DataLoader(dataset=vital_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ValinaLSTM(input_size, hidden_units, layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

hidden = (torch.randn(layers, batch_size, hidden_units, device=device),
          torch.randn(layers, batch_size, hidden_units, device=device))
train(train_loader=train_loader, model=model, epochs=epochs, optimizer=optimizer, loss_fn=criterion, device=device, summary_path='./logs/test1/', step_count=1000, hidden=hidden)
