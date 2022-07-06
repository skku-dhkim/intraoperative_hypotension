from .. import *
from src.utils import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from src.models import call_models
from datetime import datetime
# from src.utils.loss_F import call_loss_fn
# from src.utils.optimizer import call_optimizer
import torch
import ray


# TODO: Consider the deprecation.
# class TrainWrapper:
#     def __init__(self,
#                  test_loader: DataLoader,
#                  model: Module,
#                  epochs: int,
#                  optimizer: Optimizer,
#                  loss_fn,
#                  log_path: str,
#                  device: torch.device,
#                  **kwargs):
#
#         # Evaluation settings
#         self.model_path = os.path.join(log_path, 'check_points')
#         self.writer = SummaryWriter(log_dir=os.path.join(log_path, "runs"))
#
#         self.best_score = 0
#
#         self.test_acc = 0
#         self.running_loss = 0
#         self.correct = 0
#         self.scaler = torch.cuda.amp.GradScaler()
#
#         # TODO: Make it simple in future.
#         if 'step_count' in kwargs.keys():
#             self.step_count = kwargs['step_count']
#         else:
#             self.step_count = 100
#
#         if 'hidden' in kwargs.keys():
#             self.hidden_flag = kwargs['hidden']
#         else:
#             self.hidden_flag = False
#
#         if 'lr_scheduler' in kwargs.keys():
#             self.lr_scheduler = kwargs['lr_scheduler']
#         else:
#             self.lr_scheduler = None
#         ############################################################
#
#         # Model settings
#         self.device = device
#         self.model = model
#
#         self.model.to(device)
#
#         # Training settings
#         self.epochs = epochs
#         self.optimizer = optimizer
#         self.loss_fn = loss_fn
#
#         # Test settings
#         self.test_loader = test_loader
#         self.tl_counter = 0         # Train loader counter
#         self.step_counter = 0
#
#
#     def fit(self, train_loader: DataLoader, epoch: int) -> None:
#         """
#
#         Args:
#             train_loader: (DataLoader) Data loader of train files.
#             epoch: (int) Represent current epoch. Not epochs in total.
#
#         Returns: None
#
#         """
#         self.model.train()
#         pbar = tqdm(train_loader, desc="Training Steps", position=0)#steps=pbar.total
#         #if isinstance(self.lr_scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
#         #    self.lr_scheduler.T_0 = pbar.total
#
#         for x, y in pbar:
#             input_x = x.to(self.device)
#             target_y = y.to(self.device)
#
#             if self.hidden_flag:
#                 hidden = self.model.init_hidden(input_x.shape[0], self.device)
#                 predicted = self.model(input_x, hidden)
#             else:
#                 predicted = self.model(input_x)
#
#             # Get loss and back propagation
#
#             loss = self.loss_fn(predicted, target_y)
#
#             if torch.isfinite(loss):
#                 self.optimizer.zero_grad()
#                 self.scaler.scale(loss).backward()
#                 self.scaler.unscale_(self.optimizer)
#                 #loss.backward()##origin
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
#                 self.scaler.step(self.optimizer)
#
#                 #self.optimizer.step()##origin
#                 self.scaler.update()
#
#             elif not torch.isnan(loss):
#                 self.optimizer.zero_grad()
#                 self.scaler.scale(loss).backward()
#                 self.scaler.unscale_(self.optimizer)
#                 #loss.backward()##origin
#                 torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
#                 self.scaler.step(self.optimizer)
#
#                 #self.optimizer.step()##origin
#                 self.scaler.update()
#
#
#             # Training evaluation
#             self.step_counter += 1
#
#
#
#             self.running_loss += loss.item()
#             _, predicted = torch.max(predicted.data, 1)
#
#             self.correct += (predicted == target_y).sum().item()
#
#             pbar.update(1)
#             pbar.set_postfix({'Epochs': epoch, "loss": loss.item()})
#
#             if self.step_counter % self.step_count == 0:#every 1000 step
#                 self.running_loss = self.running_loss / self.step_count
#                 self.writer.add_scalar('Loss/train', self.running_loss, self.step_counter)
#                 if isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#                     self.lr_scheduler.step(self.running_loss)
#                 self.running_loss = 0
#
#             if self.lr_scheduler:
#                 if isinstance(self.lr_scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
#                     self.lr_scheduler.step()
#
#
#         if self.lr_scheduler:
#             if not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#                 if not isinstance(self.lr_scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
#                     self.lr_scheduler.step()
#
#
#     def evaluation(self, epoch: int) -> None:
#         """
#         Args:
#             epoch: (int) Represent current epoch. Not epochs in total.
#
#         Returns: None
#         """
#         self.model.eval()
#         score, self.test_acc = self.test()
#
#         if not os.path.exists(self.model_path):
#             os.makedirs(self.model_path)
#
#         if score > self.best_score:
#             torch.save({
#                 'epoch': epoch,
#                 'state_dict': self.model.state_dict(),
#                 'optimizer': self.optimizer.state_dict()
#             }, os.path.join(self.model_path, 'best-model-{}.pt'.format(epoch)))
#             self.best_score = score
#
#         self.writer.add_scalar('AUC Score', score, epoch)
#         self.writer.add_scalar('Test Accuracy', self.test_acc, epoch)
#
#     def test(self) -> tuple([float, float]):
#         """
#         Returns:
#             Tuple[float, float]: AUC Score and Accuracy of test set.
#         """
#         self.model.eval()
#         pbar = tqdm(self.test_loader, desc="Test steps", position=1)
#         with torch.no_grad():
#             total_correct = 0
#             total_len = 0
#
#             list_y = []
#             list_y_hat = []
#
#             for x, y in pbar:
#                 x = x.to(self.device)
#                 y = y.to(self.device)
#
#                 if self.hidden_flag:
#                     hidden = self.model.init_hidden(x.shape[0], self.device)
#                     predicted = self.model(x, hidden)
#                     predict_prob = torch.softmax(predicted, dim=1)
#                     _, predicted_y = torch.max(predict_prob, 1)
#                 else:
#                     predicted = self.model(x)
#                     predict_prob = torch.softmax(predicted, dim=1)
#                     _, predicted_y = torch.max(predict_prob, 1)
#
#                 correct = (predicted_y == y).sum().item()
#                 accuracy = (correct / len(y)) * 100
#
#                 list_y += y.detach().cpu().tolist()
#                 list_y_hat += predicted_y.detach().cpu().tolist()
#
#                 total_correct += correct
#                 total_len += len(y)
#
#                 pbar.set_postfix({"Test Accuracy": accuracy})
#
#         total_accuracy = (total_correct / total_len) * 100
#         total_score = roc_auc_score(list_y, list_y_hat)
#
#         pbar.write("AUC score[{:.2f}] / Accuracy: {:.2f}".format(total_score, total_accuracy))
#
#         return total_score, total_accuracy


@ray.remote(num_gpus=1)
class TrainActor:
    def __init__(self,
                 model_settings: dict,
                 hyper_params: dict,
                 train_settings: dict,
                 log_path: str,
                 pid: int):

        # NOTE: Train Settings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = call_models(model_name=model_settings['model_name'],
                                 features=model_settings['features'],
                                 embedding_dim=model_settings['embedding_dim'],
                                 sequences=model_settings['sequences'],
                                 num_of_heads=model_settings['num_of_heads'],
                                 chunk_size=model_settings['chunk_size'],
                                 hop_size=model_settings['hop_size'],
                                 hidden_channels=model_settings['hidden_channels'],
                                 num_layers=model_settings['num_layers'],
                                 low_dimension=model_settings['low_dimension'],
                                 linear=model_settings['linear']).to(self.device)
        self.optimizer = call_optimizer(optim_name=train_settings['optimizer'])(self.model.parameters(), lr=hyper_params['lr'])
        self.loss_fn = call_loss_fn(train_settings['loss_fn']).to(self.device)##grad_clip??
        self.clip = train_settings['clip']

        # Evaluation settings
        dt = datetime.now().strftime("%d-%m-%Y_%H:%M")
        self.log_path = os.path.join(log_path, "{}_{}_{}_{}_{}".format(model_settings['model_name'],
                                                                       train_settings['loss_fn'],
                                                                       train_settings['optimizer'],
                                                                       dt,
                                                                       pid))

        self.model_path = os.path.join(self.log_path, 'check_points')
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, 'runs'))

        self.save_count = train_settings['save_count']
        self.step_counter = 0

        self.best_score = 0
        self.running_loss = 0
        self.correct = 0

        self.model_settings = model_settings
        self.train_setting = train_settings
        self.hyper_params = hyper_params

        self.scaler = torch.cuda.amp.GradScaler()

    async def fit(self, x, y) -> None:
        input_x = x.to(self.device)
        target_y = y.to(self.device)

        predicted = self.model(input_x)

        # Get loss and back propagation
        loss = self.loss_fn(predicted, target_y)

        # TODO: Dongwon, please check yourself
        # Backpropagation
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        if self.clip:
            if torch.isfinite(loss):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
            elif not torch.isnan(loss):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Training evaluation
        self.step_counter += 1

        self.running_loss += loss.item()
        _, predicted = torch.max(predicted.data, 1)

        self.correct += (predicted == target_y).sum().item()

        if self.step_counter % self.save_count == 0:    # every N step
            self.running_loss = self.running_loss / self.save_count
            self.writer.add_scalar('Loss/train', self.running_loss, self.step_counter)
            self.running_loss = 0

        # if self.lr_scheduler:
        #     if isinstance(self.lr_scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
        #         self.lr_scheduler.step()

    # if self.lr_scheduler:
    #     if not isinstance(self.lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
    #         if not isinstance(self.lr_scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
    #             self.lr_scheduler.step()
    #
    def evaluation(self, test_loader, epoch) -> None:
        """
        Args:
            epoch: (int) Represent current epoch. Not epochs in total.

        Returns: None
        """
        score, test_acc = self.test(test_loader)

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
        self.writer.add_scalar('Test/Accuracy', test_acc, epoch)

    def test(self, test_loader):
        """
        Returns:
            Tuple[float, float]: AUC Score and Accuracy of test set.
        """
        pbar = tqdm(test_loader, desc="Test steps")
        with torch.no_grad():
            total_correct = 0
            total_len = 0

            list_y = []
            list_y_hat = []

            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)

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

    def write_logs(self):
        txt_path = os.path.join(self.log_path, "result_summary.txt")

        with open(txt_path, "w+") as file:
            for k, v in self.model_settings.items():
                file.write("{}: {}\n".format(k, v))
            for k, v in self.train_setting.items():
                file.write("{}: {}\n".format(k, v))
            for k, v in self.hyper_params.items():
                file.write("{}: {}\n".format(k, v))
            file.write("AUC Score: {}\n".format(self.best_score))

    def __del__(self):
        self.writer.close()
