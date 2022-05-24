from . import *

import torch
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


class TrainWrapper:
    def __init__(self,
                 test_loader: DataLoader,
                 model: Module,
                 epochs: int,
                 optimizer: Optimizer,
                 loss_fn,
                 log_path: str,
                 device: torch.device,
                 **kwargs):

        # Evaluation settings
        self.model_path = os.path.join(log_path, 'check_points')
        self.writer = SummaryWriter(log_dir=os.path.join(log_path, "runs"))

        self.best_score = 0

        self.test_acc = 0
        self.running_loss = 0
        self.correct = 0

        # TODO: Make it simple in future.
        if 'step_count' in kwargs.keys():
            self.step_count = kwargs['step_count']
        else:
            self.step_count = 1000

        if 'hidden' in kwargs.keys():
            self.hidden_flag = kwargs['hidden']
        else:
            self.hidden_flag = False

        if 'lr_scheduler' in kwargs.keys():
            self.lr_scheduler = kwargs['lr_scheduler']
        else:
            self.lr_scheduler = None
        ############################################################

        # Model settings
        self.device = device
        self.model = model
        self.model.to(device)

        # Training settings
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        # Test settings
        self.test_loader = test_loader
        self.tl_counter = 0         # Train loader counter
        self.step_counter = 0

    def fit(self, train_loader: DataLoader, epoch: int) -> None:
        """

        Args:
            train_loader: (DataLoader) Data loader of train files.
            epoch: (int) Represent current epoch. Not epochs in total.

        Returns: None

        """
        pbar = tqdm(train_loader, desc="Training Steps", position=0)
        for x, y in pbar:
            input_x = x.to(self.device)
            target_y = y.to(self.device)

            if self.hidden_flag:
                hidden = self.model.init_hidden(input_x.shape[0], self.device)
                predicted = self.model(input_x, hidden)
            else:
                predicted = self.model(input_x)

            # Get loss and back propagation
            loss = self.loss_fn(predicted, target_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Training evaluation
            self.step_counter += 1

            self.running_loss += loss.item()
            _, predicted = torch.max(predicted.data, 1)

            self.correct += (predicted == target_y).sum().item()

            pbar.update(1)
            pbar.set_postfix({'Epochs': epoch, "loss": loss.item()})

            if self.step_counter % self.step_count == 0:
                self.running_loss = self.running_loss / self.step_count
                self.writer.add_scalar('Loss/train', self.running_loss, self.step_counter)
                if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(self.running_loss)
                self.running_loss = 0

        if self.lr_scheduler:
            if not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step()

    def evaluation(self, epoch: int) -> None:
        """
        Args:
            epoch: (int) Represent current epoch. Not epochs in total.

        Returns: None
        """
        score, self.test_acc = self.test()

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if score > self.best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.model_path, 'best-model-{}.pt'.format(epoch)))
            self.best_score = score

        self.writer.add_scalar('AUC Score', score, epoch)
        self.writer.add_scalar('Test Accuracy', self.test_acc, epoch)

    def test(self) -> Tuple([float, float]):
        """
        Returns:
            Tuple[float, float]: AUC Score and Accuracy of test set.
        """
        pbar = tqdm(self.test_loader, desc="Test steps", position=1)
        with torch.no_grad():
            total_correct = 0
            total_len = 0

            list_y = []
            list_y_hat = []

            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

                if self.hidden_flag:
                    hidden = self.model.init_hidden(x.shape[0], self.device)
                    predicted = self.model(x, hidden)
                    predict_prob = torch.softmax(predicted, dim=1)
                    _, predicted_y = torch.max(predict_prob, 1)
                else:
                    predicted = self.model(x)
                    predict_prob = torch.softmax(predicted, dim=1)
                    _, predicted_y = torch.max(predict_prob, 1)

                correct = (predicted_y == y).sum().item()
                accuracy = (correct / len(y)) * 100

                list_y += y.detach().cpu().tolist()
                list_y_hat += predicted_y.detach().cpu().tolist()

                total_correct += correct
                total_len += len(y)

                pbar.set_postfix({"Test Accuracy": accuracy})

        total_accuracy = (total_correct / total_len) * 100
        total_score = roc_auc_score(list_y, list_y_hat)

        pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))

        return total_score, total_accuracy
