import numpy as np
import torch
import os
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, roc_auc_score


def train(data_loader,
          test_loader,
          model,
          epochs,
          optimizer,
          loss_fn,
          summary_path,
          model_path,
          device,
          **kwargs):

    # TODO: Tensorboard need to be fixed.
    writer = SummaryWriter(log_dir=summary_path+"/runs")
    model = model.to(device)

    best_score = 0
    test_acc = 0

    if 'step_count' in kwargs.keys():
        step_count = kwargs['step_count']
    else:
        step_count = 1000

    if 'hidden' in kwargs.keys():
        hidden_flag = kwargs['hidden']
    else:
        hidden_flag = False

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

            if hidden_flag:
                hidden = model.init_hidden(input_x.shape[0], device)
                predicted = model(input_x, hidden)
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
                writer.add_scalar('Loss/train', running_loss, epoch*step_counter+step_counter)
                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(running_loss)
                running_loss = 0

        if lr_scheduler:
            if not isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step()

        score, test_acc = test(test_loader, model, device=device, hidden=hidden_flag)

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        if score > best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_path+'/best-model-{}.pt'.format(epoch))
        else:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict()
            }, model_path + '/model-{}.pt'.format(epoch))

        accuracy = correct/len(data_loader.dataset) * 100
        pbar.write("Epoch[{}] - Train Accuracy: {:.2f}".format(epoch, accuracy))
        pbar.write("Test_acc[{:.2f}] - Score:[{:.2f}]".format(test_acc, score))
        pbar.close()

        writer.add_scalar('Score', score)
        writer.add_scalar('Accuracy', accuracy, epoch)
    writer.close()
    return model, best_score, test_acc


def test(data_loader, model, device, **kwargs):
    if 'hidden' in kwargs.keys():
        hidden_flag = kwargs['hidden']
    else:
        hidden_flag = False

    pbar = tqdm(data_loader, desc="Test steps")

    model.to(device)
    with torch.no_grad():
        total_correct = 0

        total_len = 0
        total_score = 0
        batch_len = len(pbar)

        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            if hidden_flag:
                hidden = model.init_hidden(x.shape[0], device)
                predicted = model(x, hidden)
                predict_prob = torch.softmax(predicted, dim=1)
                _, predicted_y = torch.max(predict_prob, 1)
            else:
                predicted = model(x)
                predict_prob = torch.softmax(predicted, dim=1)
                _, predicted_y = torch.max(predict_prob, 1)

            correct = (predicted_y == y).sum().item()
            accuracy = (correct/len(y)) * 100

            total_correct += correct
            total_len += len(y)

            score = roc_auc_score(y.detach().cpu().tolist(), predict_prob.detach().cpu().tolist(), multi_class='ovr')
            total_score += score

            pbar.set_postfix({"Accuracy": accuracy, "AUROC Score": score})

    total_score = total_score/batch_len
    total_accuracy = (total_correct/total_len)*100

    pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))

    return total_score, total_accuracy
