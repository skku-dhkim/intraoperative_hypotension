import torch
from torch.utils.tensorboard import SummaryWriter
from src.utils.tqdm_handler import TqdmLoggingHandler
import logging
from tqdm import tqdm


def train(train_loader, model, epochs, optimizer, loss_fn, device, summary_path, **kwargs):
    writer = SummaryWriter(log_dir=summary_path)
    log = logging.getLogger(__name__)
    log.setLevel(logging.INFO)
    log.addHandler(TqdmLoggingHandler())

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        if 'step_count' in kwargs.keys():
            step_count = kwargs['step_count']
        else:
            step_count = 1000
        if 'hidden' in kwargs.keys():
            hidden = kwargs['hidden']
        else:
            hidden = None
        step_counter = 0
        running_loss = 0
        for x, y in tqdm(train_loader, desc="Training steps"):
            input_x = x.to(device)
            target_y = y.to(device)
            target_y = target_y.to(torch.float)
            target_y = torch.reshape(target_y, (target_y.shape[0], 1))

            if hidden:
                output = model(hidden, input_x)
            else:
                output = model(input_x)

            loss = loss_fn(output, target_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_counter += 1
            running_loss += loss.item()

            if step_counter % step_count == 0:
                train_acc = torch.sum(output == target_y)
                running_loss = running_loss / step_count
                log.info("Epochs [{}]: ACC: {}, loss:{}".format(epoch, train_acc, running_loss))
                writer.add_scalar('Loss/train', running_loss, epoch*step_counter+step_counter)
                writer.add_scalar('Accuracy/train', train_acc, epoch*step_counter+step_counter)
                running_loss = 0
