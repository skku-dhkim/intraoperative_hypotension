import torch
import argparse

from torch.utils.data import DataLoader
from src.utils.data_loader import VitalDataset, load_from_hdf5
from src.train.train_function import test
from src.models.rnn import ValinaLSTM
from src.models.cnn import OneDimCNN, MultiChannelCNN, AttentionCNN, MultiHeadAttentionCNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # NOTE: Data path argument
    parser.add_argument("--test_data", type=str, required=True)

    # NOTE: Training setting argument
    parser.add_argument("--test_batch", type=int, default=-1)

    # NOTE: Model setting argument
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attention_dim", type=int, default=128)
    parser.add_argument("--num_headers", type=int, default=8)
    parser.add_argument("--layers", type=int, default=1)

    # NOTE: Log setting argument
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--summary_step", type=int, default=1000)

    args = parser.parse_args()

    test_x, test_y = load_from_hdf5(args.test_data, dtype='test')
    print("Test Data x shape: {}".format(test_x.shape))
    print("Test Data y shape: {}".format(test_y.shape))

    # NOTE: Data setting
    test_dataset = VitalDataset(x_tensor=test_x, y_tensor=test_y)
    if args.test_batch <= 0:
        test_loader = DataLoader(dataset=test_dataset, batch_size=test_x.shape[0])
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch)

    # NOTE: Device Setting
    device = torch.device("cpu")

    input_size = test_x.shape[-1]

    # NOTE: Model Setting
    if args.model.lower() == "lstm":
        hidden = True
        model = ValinaLSTM(input_size, args.hidden_dim, args.layers, num_of_classes=3)
    elif args.model.lower() == 'cnn':
        hidden = False
        model = OneDimCNN(input_size, num_of_classes=3)
    elif args.model.lower() == 'multi-channel-cnn':
        hidden = False
        model = MultiChannelCNN(input_size=input_size, num_of_classes=3)
    elif args.model.lower() == 'attention_cnn':
        hidden = False
        model = AttentionCNN(input_size=input_size,
                             embedding_dim=args.hidden_dim,
                             attention_dim=args.attention_dim,
                             sequences=test_x.shape[1],
                             num_of_classes=3, device=device)
    elif args.model.lower() == 'multi_head_attn':
        hidden = True
        model = MultiHeadAttentionCNN(
            input_size=input_size,
            embedding_dim=args.hidden_dim,
            attention_dim=args.attention_dim,
            num_heads=args.num_headers,
            sequences=test_x.shape[1],
            num_of_classes=3,
            device=device
        )
    else:
        raise NotImplementedError()

    weights = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(weights['state_dict'])
    model.eval()
    score, accuracy = test(data_loader=test_loader, model=model, hidden=hidden, device=device)
    print(score, accuracy)


