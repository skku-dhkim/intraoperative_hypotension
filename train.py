import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import argparse
import random

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler
from src.utils.data_loader import VitalDataset, load_from_hdf5
from src.utils.train_function import train, test
from src.models.rnn import ValinaLSTM
from src.models.cnn import OneDimCNN, MultiChannelCNN, AttentionCNN, MultiHeadAttentionCNN
from src.utils.data_loader import ImbalancedDatasetSampler
from src.utils.loss_F import FocalLoss, WeightedFocalLoss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # NOTE: Data path argument
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)

    # NOTE: Training setting argument
    parser.add_argument("--train_batch", type=int, default=256)
    parser.add_argument("--test_batch", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr_scheduler", type=str)

    # NOTE: Model setting argument
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--num_headers", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)

    # NOTE: Log setting argument
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--summary_step", type=int, default=1000)

    args = parser.parse_args()

    # NOTE: Data loading
    train_x, train_y = load_from_hdf5(args.train_data, dtype='train')
    print("Data x shape: {}".format(train_x.shape))
    print("Data y shape: {}".format(train_y.shape))

    test_x, test_y = load_from_hdf5(args.test_data, dtype='test')
    print("Test Data x shape: {}".format(test_x.shape))
    print("Test Data y shape: {}".format(test_y.shape))

    # NOTE: Data setting
    vital_dataset = VitalDataset(x_tensor=train_x, y_tensor=train_y)
    test_dataset = VitalDataset(x_tensor=test_x, y_tensor=test_y)

    train_loader = DataLoader(dataset=vital_dataset, batch_size=args.train_batch,
                              sampler=ImbalancedDatasetSampler(vital_dataset))
    # train_loader = DataLoader(dataset=vital_dataset, batch_size=args.train_batch,
    #                           sampler=RandomSampler(vital_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch,
                             sampler=ImbalancedDatasetSampler(test_dataset))
    # test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch,
    #                          sampler=RandomSampler(test_dataset))

    # NOTE: Device Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = train_x.shape[-1]

    # NOTE: Model Setting
    if args.model.lower() == "lstm":
        hidden = True
        model = ValinaLSTM(input_size, args.hidden_dim, args.layers, num_of_classes=2)
    elif args.model.lower() == 'cnn':
        hidden = False
        model = OneDimCNN(input_size, num_of_classes=2)
    elif args.model.lower() == 'multi-channel-cnn':
        hidden = False
        model = MultiChannelCNN(input_size=input_size, num_of_classes=2)
    elif args.model.lower() == 'attention_cnn':
        hidden = False
        model = AttentionCNN(input_size=input_size,
                             embedding_dim=args.hidden_dim,
                             attention_dim=args.attention_dim,
                             sequences=train_x.shape[1],
                             num_of_classes=2, device=device)
    elif args.model.lower() == 'multi_head_attn':
        hidden = True
        model = MultiHeadAttentionCNN(
            input_size=input_size,
            embedding_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            num_heads=args.num_headers,
            sequences=train_x.shape[1],
            num_of_classes=2,
            device=device
        )
    else:
        raise NotImplementedError()

    # NOTE: Loss
    criterion = nn.CrossEntropyLoss()
    # criterion = WeightedFocalLoss()

    # NOTE: Optimizer settings
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError()

    # NOTE: Learning rate scheduler
    if args.lr_scheduler is None:
        lr_sched = None
    elif args.lr_scheduler.lower() == 'exponential':
        lr_sched = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
    elif args.lr_scheduler.lower() == 'step_decay':
        if not isinstance(optimizer, optim.SGD):
            raise ValueError("Optimizer should be SGD.")
        lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    else:
        raise NotImplementedError()

    # NOTE: Train wrapper
    log_path = './logs/{}_{}'.format(args.model, random.randint(0, 100))
    print("Log path: {}".format(log_path))
    model, best_score, test_acc = train(data_loader=train_loader,
                                        test_loader=test_loader,
                                        model=model,
                                        epochs=args.epochs,
                                        optimizer=optimizer,
                                        loss_fn=criterion,
                                        summary_path=log_path,
                                        step_count=args.summary_step,
                                        model_path=log_path + "/checkpoint/",
                                        device=device,
                                        hidden=hidden,
                                        lr_scheduler=lr_sched)

    with open("{}/{}_train_result.txt".format(log_path, args.model), "w") as file:
        file.write("Train data: {}\n".format(args.train_data))
        file.write("Test data: {}\n".format(args.test_data))

        file.write("Train batch: {}\n".format(args.train_batch))
        file.write("Test batch: {}\n".format(args.test_batch))
        file.write("Epochs: {}\n".format(args.epochs))
        file.write("Learning rate: {}\n".format(args.lr))
        file.write("Hidden Dim: {}\n".format(args.hidden_dim))
        file.write("Attention Dim: {}\n".format(args.attention_dim))
        file.write("Num headers: {}\n".format(args.num_headers))
        file.write("Layers: {}\n".format(args.layers))

        file.write("Accuracy: {}\n".format(test_acc))
        file.write("AUC Score: {}\n".format(best_score))
