import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from src.utils.data_loader import VitalDataset
from src.utils.train_function import train, test
from src.models.rnn import ValinaLSTM

batch_size = 128
input_size = 7
hidden_units = 256
layers = 1
epochs = 5
model_name = 'LSTM'
log_path = './logs/test1'

# f = h5py.File('./data/dataset/train_2021-12-17-17:43.hdf5', 'r')
#
# data_x = f['train']['x']
# data_y = f['train']['y']
# print("Data x shape: {}".format(data_x.shape))
# print("Data y shape: {}".format(data_y.shape))

# NOTE: This code for testing in MAC
sequences = 60
data_x = np.random.rand(1000, sequences, input_size)
data_y = np.random.randint(3, size=1000)
print(data_x.shape)
print(data_y.shape)
#
vital_dataset = VitalDataset(x_tensor=data_x, y_tensor=data_y)
train_loader = DataLoader(dataset=vital_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=vital_dataset, batch_size=data_x.shape[0], shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ValinaLSTM(input_size, hidden_units, layers, num_of_classes=3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
lr_sched = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

model = train(data_loader=train_loader,
              model=model,
              epochs=epochs,
              optimizer=optimizer,
              loss_fn=criterion,
              summary_path=log_path,
              step_count=10,
              model_path=log_path+"/checkpoint/",
              hidden=True,
              lr_scheduler=lr_sched)

score = test(data_loader=test_loader, model=model, hidden=True)
print("AUC score: {}".format(score))


with open("{}/{}_train_result.txt".format(log_path, model_name), "w") as file:
    file.write("Batch_size: {}\n".format(batch_size))
    file.write("Input_size: {}\n".format(input_size))
    file.write("Hidden_units: {}\n".format(hidden_units))
    file.write("Layers: {}\n".format(layers))
    file.write("Epochs: {}\n".format(epochs))
    file.write("AUC Score: {}\n".format(score))

