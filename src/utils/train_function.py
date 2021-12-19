import torch
from torch.utils.tensorboard import SummaryWriter


def train(train_loader, model, epochs, optimizer, loss_fn, summary_path, **kwargs):
    writer = SummaryWriter(log_dir=summary_path)
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        print(kwargs.keys())
        step_counter = 0
        running_loss = 0

        for x, y in train_loader:
            input_x = x.to(device)
            target_y = y.to(device)
            target_y = target_y.to(torch.float)
            target_y = torch.reshape(target_y, (128, 1))

            output = model(hidden, input_x)
            loss = loss_fn(output, target_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_counter += 1
            running_loss += loss.item()

            if step_count % 1000 == 0:
                train_acc = torch.sum(output == target_y)
                running_loss = running_loss / 1000
                print("Epochs [{}]: ACC: {}, loss:{}".format(epoch, train_acc, running_loss))
                writer.add_scalar('Loss/train', running_loss, epoch*step_counter+step_counter)
                writer.add_scalar('Accuracy/train', train_acc, epoch*step_counter+step_counter)
                running_loss = 0
