import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os


def train(data_loader,
          model,
          epochs,
          optimizer,
          loss_fn,
          summary_path,
          model_path,
          **kwargs):

    # TODO: Tensorbaord need to be fixed.
    writer = SummaryWriter(log_dir=summary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_acc = 0

    if 'step_count' in kwargs.keys():
        step_count = kwargs['step_count']
    else:
        step_count = 1000
    if 'hidden' in kwargs.keys():
        hidden = kwargs['hidden']
    else:
        hidden = False

    for epoch in range(epochs):
        step_counter = 0
        running_loss = 0
        correct = 0

        pbar = tqdm(data_loader, desc="Training steps")

        for x, y in pbar:
            input_x = x.to(device)
            target_y = y.to(device)

            if hidden:
                hidden = model.init_hidden(input_x.shape[0], device)
                predicted = model(hidden, input_x)
            else:
                predicted = model(input_x)

            loss = loss_fn(predicted, target_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_counter += 1

            running_loss += loss.item()
            _, predicted = torch.max(predicted.data, 1)

            correct += (predicted == target_y).sum().item()

            pbar.update(1)
            if step_counter % step_count == 0:
                running_loss = running_loss / step_count
                pbar.set_postfix({'Epochs': epoch, "loss": running_loss})
                writer.add_scalar('Loss/train', running_loss, epoch*step_counter+step_counter)
                running_loss = 0

        accuracy = correct/len(data_loader.dataset) * 100
        pbar.write("Epoch[{}] - Accuracy: {:.2f}".format(epoch, accuracy))
        pbar.close()

        writer.add_scalar('Accuracy/train', accuracy, epoch)

        if accuracy > best_acc:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), model_path+'/best_model-{}.pt'.format(epoch))
            best_acc = accuracy

    writer.close()
