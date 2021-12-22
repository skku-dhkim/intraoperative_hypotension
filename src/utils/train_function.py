import numpy as np
import torch
import os
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc


def train(data_loader,
          test_loader,
          model,
          epochs,
          optimizer,
          loss_fn,
          summary_path,
          model_path,
          **kwargs):

    # TODO: Tensorboard need to be fixed.
    writer = SummaryWriter(log_dir=summary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_score = 0
    test_acc = 0

    if 'step_count' in kwargs.keys():
        step_count = kwargs['step_count']
    else:
        step_count = 1000

    if 'hidden' in kwargs.keys():
        hidden = kwargs['hidden']
    else:
        hidden = False

    if 'lr_scheduler' in kwargs.keys():
        lr_scheduler = kwargs['lr_scheduler']
    else:
        lr_scheduler = None

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
            pbar.set_postfix({'Epochs': epoch, "loss": loss.item()})

            if step_counter % step_count == 0:
                running_loss = running_loss / step_count
                # pbar.set_postfix({'Epochs': epoch, "loss": running_loss})
                writer.add_scalar('Loss/train', running_loss, epoch*step_counter+step_counter)
                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(running_loss)
                running_loss = 0

        if lr_scheduler:
            if not isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

        accuracy = correct/len(data_loader.dataset) * 100
        pbar.write("Epoch[{}] - Accuracy: {:.2f}".format(epoch, accuracy))
        pbar.close()

        writer.add_scalar('Accuracy/train', accuracy, epoch)

        score, test_acc = test(test_loader, model, device=device)

        if score > best_score:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(model.state_dict(), model_path+'/best_model-{}.pt'.format(epoch))
            best_score = score

    writer.close()
    return model, best_score, test_acc


def test(data_loader, model, device, **kwargs):
    if 'hidden' in kwargs.keys():
        hidden = kwargs['hidden']
    else:
        hidden = False

    pbar = tqdm(data_loader, desc="Test steps")

    model.to(device)
    with torch.no_grad():
        correct = 0
        total_len = 0
        total_score = 0
        batch_len = len(pbar)

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            if hidden:
                hidden = model.init_hidden(x.shape[0], device)
                predicted = model(hidden, x)
                _, predicted = torch.max(predicted.data, 1)
            else:
                predicted = model(x)
                _, predicted = torch.max(predicted.data, 1)

            correct += (predicted == y).sum().item()
            total_len += len(y)
            accuracy = (correct/total_len) * 100

            score = 0
            fpr, tpr, thresholds = roc_curve(y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), pos_label=0)
            score += auc(fpr, tpr)
            fpr, tpr, thresholds = roc_curve(y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), pos_label=1)
            score += auc(fpr, tpr)
            fpr, tpr, thresholds = roc_curve(y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), pos_label=2)
            score += auc(fpr, tpr)

            score = score/3
            total_score += score

            pbar.set_postfix({"Accuracy": accuracy, "Score": score})

    total_score = total_score/batch_len

    pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, accuracy))

    return score, accuracy
