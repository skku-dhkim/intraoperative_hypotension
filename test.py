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
from src.models.cnn import OneDimCNN, MultiChannelCNN, AttentionCNN
from src.utils.data_loader import ImbalancedDatasetSampler

batch_size = 512
input_size = 7
hidden_units = 128
layers = 1
hidden = True
step_count = 1000

model_name = 'Attention-CNN'
log_path = './logs/Test'

f2 = h5py.File('./data/dataset/test_2021-12-17-17:52.hdf5', 'r')

test_x = f2['test']['x']
test_y = f2['test']['y']
print("Test Data x shape: {}".format(test_x.shape))
print("Test Data y shape: {}".format(test_y.shape))

test_dataset = VitalDataset(x_tensor=test_x, y_tensor=test_y)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                         sampler=ImbalancedDatasetSampler(test_dataset))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ValinaLSTM(input_size, hidden_units, layers, num_of_classes=3)
# # model = OneDimCNN(input_size=input_size, num_of_classes=3)
# # model = MultiChannelCNN(input_size=input_size, num_of_classes=3)
# model = AttentionCNN(input_size=input_size,
#                      embedding_dim=hidden_units,
#                      attention_dim=64,
#                      sequences=sequences,
#                      num_of_classes=3, device=device)

model.load_state_dict(torch.load('{}/best_model-9.pt'.format(log_path), map_location=device))
score, accuracy = test(data_loader=test_loader, model=model, hidden=hidden, device=device)


with open("{}/{}_test_result.txt".format(log_path, model_name), "w") as file:
    file.write("Input_size: {}\n".format(input_size))
    file.write("Hidden_units: {}\n".format(hidden_units))
    file.write("Layers: {}\n".format(layers))
    file.write("Accuracy: {}\n".format(accuracy))
    file.write("AUC Score: {}\n".format(score))

