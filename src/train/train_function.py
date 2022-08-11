from .. import *
from src.utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from src.models import call_models
from datetime import datetime
from torch.utils.data import DataLoader
from src.train.analyzer import Attn_Saver
import torch


class TrainActor:
    def __init__(self,
                 train_loader: DataLoader,
                 model_settings: dict,
                 train_settings: dict,
                 log_path: str,
                 valid_loader: Optional[DataLoader] = None,
                 test_loader: Optional[DataLoader] = None):


        # INFO: Data Loader settings
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # INFO: Train Settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = call_models(**model_settings).to(self.device).type(torch.float64)
        self.optimizer = call_optimizer(optim_name=train_settings['optimizer'])(self.model.parameters(),
                                                                                lr=train_settings['lr'])
        self.loss_fn = call_loss_fn(train_settings['loss_fn'], **train_settings).to(self.device)

        # INFO: Evaluation settings
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M")
        self.log_path = os.path.join(log_path, "{}_{}_{}".format(model_settings['model_name'],
                                                                 train_settings['loss_fn'], dt))

        self.model_path = os.path.join(self.log_path, 'check_points')

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, 'runs'))

        self.save_count = train_settings['save_count']

        self.best_score = 0
        self.running_loss = 0
        self.correct = 0

        # INFO: Model settings
        self.model_settings = model_settings
        self.train_setting = train_settings

        # INFO: Attention evaluations
        if train_settings['save_attn']:
            self.attn = Attn_Saver(self.log_path)
        else:
            self.attn = None

    def fit(self) -> None:
        """
        Train the model with predefined settings.
        Returns:

        """

        scaler = torch.cuda.amp.GradScaler(enabled=True)

        step_counter = 0
        correct = []
        total = []

        for epoch in tqdm(range(self.train_setting['epochs']), desc="Epochs", position=0):
            with tqdm(self.train_loader, desc="Training steps", position=1, postfix={"Train Accuracy": 0}) as pbar:
                self.model.train()

                for x, y in self.train_loader:
                    input_x = x.to(self.device).type(torch.float64)
                    target_y = y.to(self.device)

                    with torch.cuda.amp.autocast(enabled=True):
                        if self.attn:
                            predicted, _ = self.model(input_x)
                        else:
                            predicted = self.model(input_x)
                        loss = self.loss_fn(predicted, target_y)

                    # Backpropagation
                    # INFO: Calculate the loss and update the model.
                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    # scaler.unscale_(self.optimizer)
                    #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.train_setting['clip_max_norm'])

                    if self.train_setting['optimizer'] == 'lbfgs':
                        scaler.step(self.optimizer, lambda: self.loss_fn(predicted, target_y))
                    else:
                        scaler.step(self.optimizer)

                    scaler.update()

                    # Training evaluation
                    step_counter += 1

                    self.running_loss += loss.item()
                    _, predicted = torch.max(predicted.data, 1)

                    correct.append((predicted == target_y).sum().item())
                    total.append(len(target_y))

                    if step_counter % self.save_count == 0:  # every N step
                        self.running_loss = self.running_loss / self.save_count
                        self.writer.add_scalar('Loss/Train', self.running_loss, step_counter)
                        self.running_loss = 0
                    pbar.update(1)

                train_acc = sum(correct) / sum(total)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                score, acc, _ = self.evaluation(epoch)
                if self.test_loader:
                    test_score, test_acc, _ = self.test(self.test_loader, epoch)
                    pbar.set_postfix({"Train Accuracy": train_acc,
                                      "Valid Accuracy": acc, "Valid AUC": score,
                                      "Test Accuracy": test_acc, "Test AUC": test_score})
                else:
                    pbar.set_postfix({"Train Accuracy": train_acc,  "Valid Accuracy": acc, "Valid AUC": score})

                # if self.clip:
                #     if torch.isfinite(loss):
                #         #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)  # 500,5 ratio
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
                #     elif not torch.isnan(loss):
                #         #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)  # 50,10
                #         torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                #     elif torch.isnan(loss):
                #         print('error nan')

    def evaluation(self, epoch, attention_map=False):
        """
        Args:
            epoch: (int) Represent current epoch. Not epochs in total.

        Returns: None
        """
        # TODO: Maybe printing the attention map mechanism is needed.
        score, acc, losses = self._run_eval(self.valid_loader)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if score > self.best_score:
            torch.save({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.model_path, 'best-model-{}.pt'.format(epoch)))
            self.best_score = score

        self.writer.add_scalar('AUC/Valid', score, epoch)
        self.writer.add_scalar('Accuracy/Valid', acc, epoch)
        self.writer.add_scalar('Loss/Valid', losses, epoch)
        return score, acc, losses

    def test(self, data_loader: DataLoader, epoch: int = 0):
        score, test_acc, losses = self._run_eval(data_loader)
        if self.attn:
            self.attn.whole_process()
            #attn.save_random_attn(1,1)
        self.writer.add_scalar('AUC/Test', score, epoch)
        self.writer.add_scalar('Accuracy/Test', test_acc, epoch)
        self.writer.add_scalar('Loss/Test', losses, epoch)
        return score, test_acc, losses

    def _run_eval(self, data_loader: DataLoader):
        """
        Returns:
            Tuple[float, float]: AUC Score and Accuracy of test set.
        """
        self.model.eval()
        with torch.no_grad():
            total_correct = []
            total_len = []
            total_loss = []
            list_y = []
            list_y_hat = []

            with tqdm(data_loader, desc="Evaluation steps") as pbar:
                for x, y in data_loader:
                    x = x.to(self.device).type(torch.float64)
                    y = y.to(self.device)

                    if self.attn: ## x list y list
                        predicted, gatt = self.model(x)
                    else:
                        predicted = self.model(x)

                    total_loss.append(self.loss_fn(predicted, y))

                    predict_prob = torch.softmax(predicted, dim=1)
                    _, predicted_y = torch.max(predict_prob, 1)

                    total_correct.append((predicted_y == y).sum().item())
                    total_len.append(len(y))

                    list_y += y.detach().cpu().tolist()
                    list_y_hat += predicted_y.detach().cpu().tolist()

                    accuracy = sum(total_correct)/sum(total_len)
                    pbar.set_postfix({"Accuracy": accuracy})
                    pbar.update(1)
                    if self.attn:
                        self.attn.preparing(y, predicted_y, x, gatt)

        total_accuracy = sum(total_correct)/sum(total_len)
        total_score = roc_auc_score(list_y, list_y_hat)
        total_loss = sum(total_loss)/sum(total_len)

        pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))

        return total_score, total_accuracy, total_loss