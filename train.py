import argparse
import ray

from torch.utils.data import DataLoader
from src.utils.data_loader import load_files, HDF5_VitalDataset
from src.train.train_function import TrainActor
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # NOTE: Data path argument
    parser.add_argument("--dset_path", type=str, required=True)

    # TODO: Deprecate in the Future.
    # # NOTE: Training setting argument
    # parser.add_argument("--train_batch", type=int, default=512)
    # parser.add_argument("--test_batch", type=int, default=512)

    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--optimizer", type=str, default="adam")
    # parser.add_argument("--momentum", type=float, default=0.9)
    # parser.add_argument("--lr_scheduler", type=str)

    # # TODO: Please add everything you need.
    # # NOTE: Model setting argument
    # parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--hidden_dim", type=int, default=256)
    # parser.add_argument("--attention_dim", type=int, default=128)
    # parser.add_argument("--num_headers", type=int, default=8)
    # parser.add_argument("--layers", type=int, default=1)
    # parser.add_argument("--loss", type=str, default="cross")
    parser.add_argument("--resume_from", type=str, default=".")

    # NOTE: Log setting argument
    # parser.add_argument("--memo", type=str, required=True)
    parser.add_argument("--log_path", type=str, default="./logs")
    parser.add_argument("--summary_step", type=int, default=1000)

    args = parser.parse_args()

    # NOTE: Data loading
    train_list, test_list = load_files(args.dset_path, 0.2)

    # NOTE: Data setting
    print("Loading the Training set from disk: {}".format(args.dset_path))
    train_dataset = HDF5_VitalDataset(train_list)

    print("Loading the Test set from dist.")
    test_dataset = HDF5_VitalDataset(test_list)

    train_settings = {
        'test_label_count': test_dataset.label_counts(),
        'save_count': args.summary_step,
        'optimizer': 'adam',
        'loss_fn': 'cross',
        'train_batch': 3,
        'test_batch': 3
    }
    model_settings = {
        'model_name': 'attention_galr',
        'features': 8,
        'embedding_dim': 3000,
        'sequences': 3000,
        'num_of_heads': 4,
        'chunk_size': 100,
        'hop_size': 100,
        'hidden_channels': 256,
        'num_layers': 3,
        'low_dimension': False
    }
    hyper_parmas = {
        'lr': 0.001,
        'epochs': 1
    }

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_settings['train_batch'], shuffle=True)
    # train_loader = DataLoader(dataset=vital_dataset, batch_size=args.train_batch,
    #                           sampler=RandomSampler(vital_dataset))
    test_loader = DataLoader(dataset=test_dataset, batch_size=train_settings['test_batch'], shuffle=False)
    print("Loading complete")

    x_shape, y_shape = test_dataset.get_shape()

    ray.init()

    # NOTE: Model Setting
    actor1 = TrainActor.remote(model_settings=model_settings,
                               train_settings=train_settings,
                               hyper_params=hyper_parmas,
                               log_path=args.log_path,
                               pid=0)

    actor2 = TrainActor.remote(model_settings=model_settings,
                               train_settings=train_settings,
                               hyper_params=hyper_parmas,
                               log_path=args.log_path,
                               pid=1)

    for epoch in tqdm(range(hyper_parmas['epochs']), desc="Epochs", position=1):
        for x, y in tqdm(train_loader, desc='Model train', position=2):
            ray.get([actor1.fit.remote(x, y), actor2.fit.remote(x, y)])

        ray.get([actor1.evaluation.remote(test_loader, epoch), actor2.evaluation.remote(test_loader, epoch)])

    ray.get([actor1.write_logs.remote(), actor2.write_logs.remote()])

    # if args.resume_from != ".":
    #     weights = torch.load(args.resume_from, map_location=device)
    #     model.load_state_dict(weights['state_dict'])

    # # NOTE: Learning rate scheduler
    # #learning rate scheduling was not efficient on adam
    # if args.lr_scheduler is None:
    #     lr_sched = None
    # elif args.lr_scheduler.lower() == 'exponential':
    #     lr_sched = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
    # elif args.lr_scheduler.lower() == 'step_decay':
    #     if not isinstance(optimizer, optim.SGD):
    #         raise ValueError("Optimizer should be SGD.")
    #     lr_sched = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # elif args.lr_scheduler.lower() == 'cosine':
    #     lr_sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0)#
    # elif args.lr_scheduler.lower() == 'warmrestart':
    #     lr_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=300, T_mult=1, eta_min=0.0)##5000#300
    # else:
    #     raise NotImplementedError()
    # # TODO: consider adding custom scheduler

    #

